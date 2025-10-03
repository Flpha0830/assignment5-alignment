import json
import logging
import os
from datetime import datetime
from pathlib import Path

from typing import List, Callable
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    load_gsm8k_data,
    evaluate_vllm,
    log_generations,
)

logging.getLogger("vllm").setLevel(logging.WARNING)


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/test.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
):
    prompts, _, answers = load_gsm8k_data(prompt_path, data_path)

    # Based on Dr. GRPO: stop when the model completes its answer
    # https://github.com/sail-sg/understand-r1-zero/blob/
    # c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
    )

    # Save results to outputs folder
    current_dir = Path(__file__).parent
    outputs_dir = current_dir.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Create filename with current datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_evaluate.jsonl"
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
    print(f"Neither (format=0, answer=0): {neither} ({neither/total_count*100:.1f}%)")
    print(f"Avg response length: {stats['avg_len']:.2f}")
    print(f"Avg length (correct): {stats['avg_len_correct']:.2f}")
    print(f"Avg length (incorrect): {stats['avg_len_incorrect']:.2f}")


if __name__ == "__main__":
    main()
