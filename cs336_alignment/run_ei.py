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


from vllm import SamplingParams

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
class EIConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    train_data_path: str = "./data/gsm8k/train.jsonl"
    eval_data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"

    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)

    ei_steps: int = 5
    ei_batch_size: int = 2048
    ei_rollouts_per_question: int = 2

    sft_steps: int = 32
    sft_batch_size: int = 8
    sft_gradient_accumulation_steps: int = 16

    seed: int = 42


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
    ei_config = EIConfig()

    current_dir = Path(__file__).parent
    outputs_dir = current_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Setup wandb
    run = wandb.init(
        project="a5-alignment-ei",
        config=asdict(ei_config),
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Initialize vLLM and sampling parameters
    vllm_model = init_vllm(
        model_id=ei_config.model_name,
        device=ei_config.eval_device,
        seed=ei_config.seed,
    )

    rollouts_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        n=ei_config.ei_rollouts_per_question,
        seed=ei_config.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initialize model, tokenizer, optimizer, scheduler
    model = AutoModelForCausalLM.from_pretrained(
        ei_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=ei_config.train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(ei_config.model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ei_config.learning_rate, betas=ei_config.betas
    )

    total_steps = ei_config.ei_steps * ei_config.sft_steps
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare data
    train_prompts, _, train_answers = load_gsm8k_data(
        ei_config.prompt_path, ei_config.train_data_path
    )
    eval_prompts, _, eval_answers = load_gsm8k_data(
        ei_config.prompt_path, ei_config.eval_data_path
    )

    random.seed(ei_config.seed)

    # Expert Iteration loop
    best_correct_cnt = 0
    for ei_step in range(ei_config.ei_steps):
        print(f"\n=== EI Step {ei_step+1}/{ei_config.ei_steps} ===")

        # Sample a batch of questions with corresponding answers
        batch_indices = random.sample(
            range(len(train_prompts)), ei_config.ei_batch_size
        )
        batch_questions = [train_prompts[i] for i in batch_indices]
        batch_answers = [train_answers[i] for i in batch_indices]

        # Sampling G outputs for each question
        load_policy_into_vllm_instance(model, vllm_model)
        results = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=batch_questions,
            answers=batch_answers,
            eval_sampling_params=rollouts_sampling_params,
        )

        # Filter for correct outputs
        sft_prompts, sft_responses = [], []
        for result in results:
            rewards = result["rewards"]["reward"]

            if rewards != 1.0:
                continue

            prompt = result["prompt"]
            response = result["generated_text"]

            sft_prompts.append(prompt)
            sft_responses.append(response)
        print(
            f"Found {len(sft_prompts)} correct outputs out of {len(results)} total generations"
        )

        if not sft_prompts:
            print("No correct outputs in this batch. Skipping SFT update.")
            continue

        # Prepare SFT dataset
        ei_dataset = SFTDataset(sft_prompts, sft_responses)
        ei_dataloader = DataLoader(
            ei_dataset,
            batch_size=ei_config.sft_batch_size,
            shuffle=True,
            collate_fn=lambda b: sft_collate(b, tokenizer),
        )

        # SFT update
        step, loss_accum, micro_steps = 0, 0, 0
        while step < ei_config.sft_steps:
            for batch in ei_dataloader:
                input_ids = batch["input_ids"].to(ei_config.train_device)
                labels = batch["labels"].to(ei_config.train_device)
                response_mask = batch["response_mask"].to(ei_config.train_device)
                token_count = response_mask.sum().clamp(min=1.0)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    log_probs = get_response_log_probs(model, input_ids, labels)[
                        "log_probs"
                    ]
                    loss, _ = sft_microbatch_train_step(
                        log_probs,
                        response_mask,
                        ei_config.sft_gradient_accumulation_steps,
                        token_count,
                    )
                loss_accum += loss.item()
                micro_steps += 1

                if micro_steps % ei_config.sft_gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    step += 1
                    lr = scheduler.get_last_lr()[0]
                    print(f"[train] step={step}, loss={loss_accum:.4f}, lr={lr:.6e}")
                    wandb.log(
                        {
                            "train/loss": loss_accum,
                            "train/lr": lr,
                            "train_step": step + ei_step * ei_config.sft_steps,
                        }
                    )
                    loss_accum = 0

                if step >= ei_config.sft_steps:
                    break

        # Evaluation after each EI step
        load_policy_into_vllm_instance(model, vllm_model)
        results = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=eval_prompts,
            answers=eval_answers,
            eval_sampling_params=eval_sampling_params,
        )

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"ei_{timestamp}_{ei_step}.jsonl"
        filepath = Path(outputs_dir) / filename

        stats = log_generations(results, filepath)

        total_count = stats["total_count"]
        correct_both = stats["correct_both"]
        format_only = stats["format_only"]
        neither = stats["neither"]

        print(f"EI Step {ei_step}")
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
                "eval_step": ei_step,
            }
        )

        if correct_both > best_correct_cnt:
            dir_name = f"ei_{timestamp}_{ei_step}_{correct_both}"
            best_correct_cnt = correct_both
            best_model_path = save_model(dir_name, model)
            print(f"New best model saved to {best_model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
