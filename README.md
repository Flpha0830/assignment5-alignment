# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Usage

### Run unit tests:

``` sh
uv run pytest -v

=========================================================== test session starts ===========================================================
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.5.0 -- /workspace/a5-alignment/.venv/bin/python
cachedir: .pytest_cache
rootdir: /workspace/a5-alignment
configfile: pyproject.toml
plugins: anyio-4.9.0
collected 31 items                                                                                                                        

tests/test_data.py::test_packed_sft_dataset PASSED                                                                                  [  3%]
tests/test_data.py::test_iterate_batches PASSED                                                                                     [  6%]
tests/test_dpo.py::test_per_instance_dpo_loss PASSED                                                                                [  9%]
tests/test_grpo.py::test_compute_group_normalized_rewards_normalize_by_std PASSED                                                   [ 12%]
tests/test_grpo.py::test_compute_group_normalized_rewards_no_normalize_by_std PASSED                                                [ 16%]
tests/test_grpo.py::test_compute_naive_policy_gradient_loss PASSED                                                                  [ 19%]
tests/test_grpo.py::test_compute_grpo_clip_loss_large_cliprange PASSED                                                              [ 22%]
tests/test_grpo.py::test_compute_grpo_clip_loss_small_cliprange PASSED                                                              [ 25%]
tests/test_grpo.py::test_compute_policy_gradient_loss_no_baseline PASSED                                                            [ 29%]
tests/test_grpo.py::test_compute_policy_gradient_loss_reinforce_with_baseline PASSED                                                [ 32%]
tests/test_grpo.py::test_compute_policy_gradient_loss_grpo_clip PASSED                                                              [ 35%]
tests/test_grpo.py::test_masked_mean_dim0 PASSED                                                                                    [ 38%]
tests/test_grpo.py::test_masked_mean_dim1 PASSED                                                                                    [ 41%]
tests/test_grpo.py::test_masked_mean_dimlast PASSED                                                                                 [ 45%]
tests/test_grpo.py::test_masked_mean_dimNone PASSED                                                                                 [ 48%]
tests/test_grpo.py::test_grpo_microbatch_train_step_grpo_clip PASSED                                                                [ 51%]
tests/test_grpo.py::test_grpo_microbatch_train_step_grpo_clip_10_steps PASSED                                                       [ 54%]
tests/test_metrics.py::test_parse_mmlu_response PASSED                                                                              [ 58%]
tests/test_metrics.py::test_parse_mmlu_response_unknown PASSED                                                                      [ 61%]
tests/test_metrics.py::test_parse_gsm8k_response PASSED                                                                             [ 64%]
tests/test_metrics.py::test_parse_gsm8k_response_unknown PASSED                                                                     [ 67%]
tests/test_sft.py::test_tokenize_prompt_and_output PASSED                                                                           [ 70%]
tests/test_sft.py::test_compute_entropy PASSED                                                                                      [ 74%]
tests/test_sft.py::test_get_response_log_probs PASSED                                                                               [ 77%]
tests/test_sft.py::test_masked_normalize_dim0 PASSED                                                                                [ 80%]
tests/test_sft.py::test_masked_normalize_dim1 PASSED                                                                                [ 83%]
tests/test_sft.py::test_masked_normalize_dimlast PASSED                                                                             [ 87%]
tests/test_sft.py::test_masked_normalize_dimNone PASSED                                                                             [ 90%]
tests/test_sft.py::test_sft_microbatch_train_step PASSED                                                                            [ 93%]
tests/test_sft.py::test_sft_microbatch_train_step_normalize PASSED                                                                  [ 96%]
tests/test_sft.py::test_sft_microbatch_train_step_10_steps PASSED                                                                   [100%]

====================================================== 31 passed in 62.90s (0:01:02) ======================================================
```

### Run evaluation:

```sh
uv run cs336_alignment/run_evaluate.py
```

### Run training scripts:

**Note**: Training scripts require 2 GPUs.

```sh
# Supervised Finetuning
uv run cs336_alignment/run_sft.py

# Expert Iteration
uv run cs336_alignment/run_ei.py

# Group Relative Policy Optimization (GRPO)
uv run cs336_alignment/run_grpo.py
```
