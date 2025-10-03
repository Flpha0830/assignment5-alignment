import json
import random
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Callable

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


from vllm import LLM, SamplingParams

from cs336_alignment.vllm_utils import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    load_gsm8k_data,
    evaluate_vllm,
    log_generations,
    save_model,
)


logging.getLogger("vllm").setLevel(logging.WARNING)


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "./data/gsm8k/train.jsonl"
    eval_data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"

    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)

    training_steps: int = 512
    eval_interval_steps: int = 32

    seed: int = 42
    num_examples: int | None = 256


class SFTDataset(Dataset):
    def __init__(self, prompts: List[str], answers: List[str]):
        self.prompts = prompts
        self.answers = answers

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.answers[idx]


def sft_collate(batch, tokenizer):
    prompts, answers = map(list, zip(*batch))
    enc = tokenize_prompt_and_output(prompts, answers, tokenizer)
    return {**enc, "answers": answers}


def main():
    sft_config = SFTConfig()

    current_dir = Path(__file__).parent
    outputs_dir = current_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Setup wandb
    run = wandb.init(
        project="a5-alignment-sft",
        config=asdict(sft_config),
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Initialize vLLM and sampling parameters
    vllm_model = init_vllm(
        model_id=sft_config.model_name,
        device=sft_config.eval_device,
        seed=sft_config.seed,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initialize model, tokenizer, optimizer, scheduler
    model = AutoModelForCausalLM.from_pretrained(
        sft_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=sft_config.train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_config.model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sft_config.learning_rate, betas=sft_config.betas
    )

    total_steps = sft_config.training_steps
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare data
    train_prompts, train_responses, _ = load_gsm8k_data(
        sft_config.prompt_path, sft_config.train_data_path
    )
    eval_prompts, _, eval_answers = load_gsm8k_data(
        sft_config.prompt_path, sft_config.eval_data_path
    )
    dataset = SFTDataset(train_prompts, train_responses)

    if sft_config.num_examples is not None:
        random.seed(sft_config.seed)
        indices = random.sample(range(len(dataset)), sft_config.num_examples)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=sft_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate(b, tokenizer),
    )

    best_correct_cnt = 0.0
    step, micro_steps, loss_accum = 0, 0, 0
    while step < sft_config.training_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(sft_config.train_device)
            labels = batch["labels"].to(sft_config.train_device)
            response_mask = batch["response_mask"].to(sft_config.train_device)
            token_count = response_mask.sum().clamp(min=1.0)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                log_probs = get_response_log_probs(model, input_ids, labels)[
                    "log_probs"
                ]
                loss, _ = sft_microbatch_train_step(
                    log_probs,
                    response_mask,
                    sft_config.gradient_accumulation_steps,
                    token_count,
                )

            loss_accum += loss.item()
            micro_steps += 1

            if micro_steps % sft_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                lr = scheduler.get_last_lr()[0]
                print(f"[train] step={step}, loss={loss_accum:.4f}, lr={lr:.6e}")
                wandb.log(
                    {"train/loss": loss_accum, "train/lr": lr, "train_step": step}
                )
                loss_accum = 0

                # Periodic evaluation
                if step % sft_config.eval_interval_steps == 0:
                    load_policy_into_vllm_instance(model, vllm_model)

                    results = evaluate_vllm(
                        vllm_model=vllm_model,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        answers=eval_answers,
                        eval_sampling_params=sampling_params,
                    )
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"sft_{timestamp}_{step}.jsonl"
                    filepath = outputs_dir / filename

                    stats = log_generations(results, filepath)

                    total_count = stats["total_count"]
                    correct_both = stats["correct_both"]
                    format_only = stats["format_only"]
                    neither = stats["neither"]

                    print(f"Total results: {total_count}")
                    print(
                        f"Correct (format=1, answer=1): {correct_both} ({correct_both/total_count*100:.1f}%)"
                    )
                    print(
                        f"Format only (format=1, answer=0): {format_only} ({format_only/total_count*100:.1f}%)"
                    )
                    print(
                        f"Neither (format=0, answer=0): {neither} ({neither/total_count*100:.1f}%)"
                    )
                    print(f"Avg response length: {stats['avg_len']:.2f}")
                    print(f"Avg length (correct): {stats['avg_len_correct']:.2f}")
                    print(f"Avg length (incorrect): {stats['avg_len_incorrect']:.2f}")

                    wandb.log(
                        {
                            "eval/correct": correct_both,
                            "eval/format_only": format_only,
                            "eval/neither": neither,
                            "eval_step": step,
                        }
                    )

                    if correct_both > best_correct_cnt:
                        dir_name = f"sft_{timestamp}_{step}_{correct_both}"
                        best_correct_cnt = correct_both
                        best_model_path = save_model(dir_name, model)
                        print(f"New best model saved to {best_model_path}")

            if step >= sft_config.training_steps:
                break

    wandb.finish()


if __name__ == "__main__":
    main()
