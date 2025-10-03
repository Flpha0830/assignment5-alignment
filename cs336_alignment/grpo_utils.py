import torch


from typing import Callable, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            "advantages": torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            "raw_rewards": torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            "metadata": dict[str, float]:
                metadata for the rewards of the rollout batch.
    """

    raw_rewards = []
    for rollout_response, ground_truth in zip(
        rollout_responses, repeated_ground_truths
    ):
        reward_dict = reward_fn(rollout_response, ground_truth)
        raw_rewards.append(reward_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards)
    raw_rewards_per_group = raw_rewards.view(-1, group_size)
    mean_rewards_per_group = raw_rewards_per_group.mean(dim=-1, keepdim=True)
    advantage = raw_rewards_per_group - mean_rewards_per_group

    if normalize_by_std:
        std_rewards_per_group = raw_rewards_per_group.std(dim=-1, keepdim=True)
        advantage /= std_rewards_per_group + advantage_eps

    advantages = advantage.view(-1)
    metadata = {
        "raw_reward_mean": raw_rewards.mean().item(),
        "raw_reward_std": raw_rewards.std().item(),
        "raw_reward_min": raw_rewards.min().item(),
        "raw_reward_max": raw_rewards.max().item(),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """

    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            "loss": torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            "metadata": dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    log_ratio = policy_log_probs - old_log_probs

    ratio = torch.exp(log_ratio)
    unclipped_loss = advantages * ratio

    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = advantages * clipped_ratio

    loss = -torch.min(unclipped_loss, clipped_loss)

    # Log whether each token was clipped (i.e., clipped loss < unclipped loss)
    was_clipped = clipped_loss < unclipped_loss
    metadata = {"was_clipped": was_clipped}

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            per-token log-probabilities from the policy being trained.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor of shape (batch_size, 1) | None:
            Required if loss_type == "no_baseline".
        advantages: torch.Tensor of shape (batch_size, 1) | None:
            Required for "reinforce_with_baseline" and "grpo_clip".
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length) | None:
            Required for "grpo_clip".
        cliprange: float | None, Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            "loss": torch.Tensor of shape (batch_size, sequence_length), per-token loss.
            "metadata": dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert (
            advantages is not None
        ), "advantages is required for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs is required for grpo_clip loss"
        assert cliprange is not None, "cliprange is required for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss, metadata
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included in the mean.
        dim: int | None, the dimension to compute the mean along.
            If None, compute the mean over all masked elements.

    Returns:
        torch.Tensor, the masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            per-token log-probabilities from the policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response. 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.

            "loss": torch.Tensor:
                a scalar tensor. The microbatch loss, adjusted for gradient accumulation.
            "metadata": dict[str, torch.Tensor]:
                Dict with metadata from the underlying loss call, and any other statistics.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(loss, response_mask, dim=-1).mean() / gradient_accumulation_steps
    loss.backward()

    return loss, metadata
