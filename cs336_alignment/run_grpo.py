import random
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Callable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from vllm import SamplingParams


from cs336_alignment.vllm_utils import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
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
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    train_data_path: str = "./data/gsm8k/train.jsonl"
    eval_data_path: str = "./data/gsm8k/test.jsonl"

    train_device: str = "cuda:0"
    eval_device: str = "cuda:1"

    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    advantage_eps: float = 1e-6
    cliprange: float = 0.2

    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    group_size: int = 8

    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = (
        "grpo_clip"
    )
    use_std_normalization: bool = True

    precompute_batch_size: int = 16
    eval_step_interval: int = 5

    seed: int = 42


class GRPODataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        raw_rewards: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask
        self.old_log_probs = old_log_probs
        self.advantages = advantages
        self.raw_rewards = raw_rewards

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "response_mask": self.response_mask[idx],
            "old_log_probs": self.old_log_probs[idx],
            "advantages": self.advantages[idx],
            "raw_rewards": self.raw_rewards[idx],
        }


def main():
    grpo_config = GRPOConfig()

    assert (
        grpo_config.train_batch_size % grpo_config.gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = (
        grpo_config.train_batch_size // grpo_config.gradient_accumulation_steps
    )
    assert (
        grpo_config.rollout_batch_size % grpo_config.group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = (
        grpo_config.rollout_batch_size // grpo_config.group_size
    )
    assert (
        grpo_config.train_batch_size >= grpo_config.group_size
    ), "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = (
        grpo_config.rollout_batch_size // micro_train_batch_size
    )

    current_dir = Path(__file__).parent
    outputs_dir = current_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Setup wandb
    run = wandb.init(
        project="a5-alignment-grpo",
        config=asdict(grpo_config),
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Initialize vLLM and sampling parameters
    vllm_model = init_vllm(
        model_id=grpo_config.model_name,
        device=grpo_config.eval_device,
        seed=grpo_config.seed,
    )

    rollouts_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=1024,
        n=grpo_config.group_size,
        seed=grpo_config.seed,
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
        grpo_config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=grpo_config.train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=grpo_config.learning_rate,
        betas=grpo_config.betas,
        weight_decay=0.0,
    )

    total_steps = (
        grpo_config.n_grpo_steps
        * grpo_config.epochs_per_rollout_batch
        * n_microbatches_per_rollout_batch
        // grpo_config.gradient_accumulation_steps
    )
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
    )

    # Prepare data
    train_prompts, _, train_answers = load_gsm8k_data(
        grpo_config.prompt_path, grpo_config.train_data_path
    )
    eval_prompts, _, eval_answers = load_gsm8k_data(
        grpo_config.prompt_path, grpo_config.eval_data_path
    )

    random.seed(grpo_config.seed)

    # start GRPO loop
    best_correct_cnt = 0
    global_step = 0
    for grpo_step in range(1, grpo_config.n_grpo_steps + 1):
        print(f"\n=== GRPO Step {grpo_step}/{grpo_config.n_grpo_steps} ===")

        # Sample a batch of questions with corresponding answers (n_prompts_per_rollout_batch)
        batch_indices = random.sample(
            range(len(train_prompts)), n_prompts_per_rollout_batch
        )
        batch_questions = [train_prompts[i] for i in batch_indices]
        batch_answers = [train_answers[i] for i in batch_indices]

        # Load old policy into vLLM and sample rollouts:
        load_policy_into_vllm_instance(model, vllm_model)
        results = evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=batch_questions,
            answers=batch_answers,
            eval_sampling_params=rollouts_sampling_params,
        )

        # 4) Collect responses
        grpo_prompts = []
        grpo_responses = []
        grpo_answers = []

        for result in results:
            prompt = result["prompt"]
            response = result["generated_text"]
            answer = result["answer"]

            grpo_prompts.append(prompt)
            grpo_responses.append(response)
            grpo_answers.append(answer)

        # Compute group-normalized advantages and raw rewards
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses=grpo_responses,
            repeated_ground_truths=grpo_answers,
            group_size=grpo_config.group_size,
            advantage_eps=grpo_config.advantage_eps,
            normalize_by_std=grpo_config.use_std_normalization,
        )

        # Debug: Print advantages and raw rewards distribution
        print(
            f"Advantages - mean: {advantages.mean():.6f}, std: {advantages.std():.6f}, min: {advantages.min():.6f}, max: {advantages.max():.6f}"
        )
        print(
            f"Raw rewards - mean: {raw_rewards.mean():.6f}, std: {raw_rewards.std():.6f}, min: {raw_rewards.min():.6f}, max: {raw_rewards.max():.6f}"
        )
        print(f"Advantages==0 count: {(advantages == 0).sum()}/{len(advantages)}")

        avg_abs_advantage = advantages.abs().mean().item()
        print(f"Average absolute advantage: {avg_abs_advantage:.8f}")

        # Precompute old log probs with old policy
        enc = tokenize_prompt_and_output(grpo_prompts, grpo_responses, tokenizer)
        grpo_input_ids = enc["input_ids"].to(grpo_config.train_device)
        grpo_labels = enc["labels"].to(grpo_config.train_device)
        grpo_response_mask = enc["response_mask"].to(grpo_config.train_device)

        # Process in smaller batches to avoid OOM
        grpo_old_log_probs_list = []

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(grpo_input_ids), grpo_config.precompute_batch_size):
                end_idx = min(
                    i + grpo_config.precompute_batch_size, len(grpo_input_ids)
                )
                batch_input_ids = grpo_input_ids[i:end_idx]
                batch_labels = grpo_labels[i:end_idx]

                batch_old_log_probs = get_response_log_probs(
                    model, batch_input_ids, batch_labels
                )["log_probs"]
                grpo_old_log_probs_list.append(batch_old_log_probs)

        grpo_old_log_probs = torch.cat(grpo_old_log_probs_list, dim=0)

        # Move to CPU for dataset storage
        grpo_input_ids = grpo_input_ids.cpu()
        grpo_labels = grpo_labels.cpu()
        grpo_response_mask = grpo_response_mask.cpu()
        grpo_old_log_probs = grpo_old_log_probs.cpu()

        # Prepare GRPO dataset
        grpo_dataset = GRPODataset(
            grpo_input_ids,
            grpo_labels,
            grpo_response_mask,
            grpo_old_log_probs,
            advantages,
            raw_rewards,
        )
        grpo_dataloader = DataLoader(
            grpo_dataset,
            batch_size=micro_train_batch_size,
            shuffle=True,
        )

        loss_accum, micro_steps = 0, 0
        for epoch in range(1, grpo_config.epochs_per_rollout_batch + 1):
            for batch in grpo_dataloader:
                input_ids = batch["input_ids"].to(grpo_config.train_device)
                labels = batch["labels"].to(grpo_config.train_device)
                response_mask = batch["response_mask"].to(grpo_config.train_device)
                old_log_probs = batch["old_log_probs"].to(grpo_config.train_device)
                advantages = batch["advantages"].to(grpo_config.train_device)
                raw_rewards = batch["raw_rewards"].to(grpo_config.train_device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    policy_log_probs = get_response_log_probs(model, input_ids, labels)[
                        "log_probs"
                    ]

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
                        loss_type=grpo_config.loss_type,
                        raw_rewards=raw_rewards.view(-1, 1),
                        advantages=advantages.view(-1, 1),
                        old_log_probs=old_log_probs,
                        cliprange=grpo_config.cliprange,
                    )

                loss_accum += loss.item()
                micro_steps += 1

                # Log individual microbatch loss for debugging
                if micro_steps % 8 == 0:
                    print(
                        f"  microbatch {micro_steps}: loss={loss.item():.8f}, loss_accum={loss_accum:.8f}"
                    )

                if micro_steps % grpo_config.gradient_accumulation_steps == 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    lr = scheduler.get_last_lr()[0]
                    wandb.log(
                        {
                            "train/loss": loss_accum,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train_step": global_step,
                        }
                    )
                    print(
                        f"[train] step={global_step}, grpo_step={grpo_step}, epoch={epoch}, loss_accum={loss_accum:.8f}, lr={lr:.6e}"
                    )

                    loss_accum = 0

        # Evaluation
        if grpo_step % grpo_config.eval_step_interval == 0:
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
            filename = f"grpo_{timestamp}_{grpo_step}.jsonl"
            filepath = Path(outputs_dir) / filename

            stats = log_generations(results, filepath)

            total_count = stats["total_count"]
            correct_both = stats["correct_both"]
            format_only = stats["format_only"]
            neither = stats["neither"]

            print(f"GRPO Step {grpo_step}")
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
                    "eval_step": grpo_step,
                }
            )

            if correct_both > best_correct_cnt:
                dir_name = f"grpo_{timestamp}_{grpo_step}_{correct_both}"
                best_correct_cnt = correct_both
                best_model_path = save_model(dir_name, model)
                print(f"New best model saved to {best_model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
