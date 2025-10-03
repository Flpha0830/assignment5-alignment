import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from typing import List


def tokenize_prompt_and_output(
    prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    assert (
        len(output_strs) == batch_size
    ), "prompt_strs and output_strs must have same length"

    pad_token_id = tokenizer.pad_token_id

    prompt_token_ids = []
    output_token_ids = []
    max_len = 0
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_tokens = tokenizer.encode(output_str, add_special_tokens=False)

        prompt_token_ids.append(torch.tensor(prompt_tokens))
        output_token_ids.append(torch.tensor(output_tokens))

        total_len = len(prompt_tokens) + len(output_tokens)
        max_len = max(max_len, total_len)

    input_ids = []
    labels = []
    response_mask = []
    for i in range(batch_size):
        prompt_tokens = prompt_token_ids[i]
        output_tokens = output_token_ids[i]
        combined_tokens = torch.cat([prompt_tokens, output_tokens], dim=0)

        pad_length = max_len - len(combined_tokens)
        padded_combined_tokens = F.pad(
            combined_tokens, (0, pad_length), value=pad_token_id
        )

        input_ids.append(padded_combined_tokens[:-1])
        labels.append(padded_combined_tokens[1:])

        padded_response_mask = F.pad(
            torch.cat(
                [torch.zeros_like(prompt_tokens), torch.ones_like(output_tokens)], dim=0
            ),
            (-1, pad_length),
            value=False,
        )
        response_mask.append(padded_response_mask)

    input_ids = torch.stack(input_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    response_mask = torch.stack(response_mask, dim=0)
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions.
    """
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    response_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if not return_token_entropy:
        return {"log_probs": response_log_probs}

    token_entropy = compute_entropy(logits)
    return {"log_probs": response_log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float, the constant to divide by
            for normalization.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, Number of microbatches per optimizer step.
        normalize_constant: float, The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            "loss": torch.Tensor:
                a scalar tensor. The microbatch loss, adjusted for gradient accumulation.
            "metadata": dict[str, torch.Tensor]:
                Dict with metadata from the underlying loss call, and any other statistics.
    """
    loss = (
        -masked_normalize(
            policy_log_probs, response_mask, normalize_constant, dim=-1
        ).mean()
        / gradient_accumulation_steps
    )

    loss.backward()

    token_count = response_mask.sum()
    metadata = {"token_count": token_count.detach()}

    return loss, metadata
