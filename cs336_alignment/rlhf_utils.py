import os
import json
import random
import re
from typing import List, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from .sft_utils import get_response_log_probs


class PackedSFTDataset(Dataset):
    """
    A PyTorch Dataset for instruction tuning that packs tokenized examples
    into sequences of a fixed length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
        prompt_template_path: str | os.PathLike,
    ):
        """
        Constructs the dataset.

        Args:
            tokenizer: A transformers tokenizer for tokenizing and encoding text
            dataset_path: Path to instruction tuning data
            seq_length: Desired length of sequences (typically model context length)
            shuffle: Whether to shuffle documents before concatenation
            prompt_template_path: Path to prompt template file.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.prompt_template = self._load_prompt_template(prompt_template_path)

        # Read and process the dataset
        examples = self._load_examples(dataset_path)
        if shuffle:
            random.shuffle(examples)

        # Tokenize and concatenate all examples
        all_tokens = self._tokenize_and_concatenate(examples)

        # Pack into sequences of seq_length
        self.sequences = self._pack_sequences(all_tokens)

    def _load_prompt_template(self, template_path: str | os.PathLike | None) -> str:
        """Load prompt template."""
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read().strip()
        return template

    def _load_examples(self, dataset_path: str | os.PathLike) -> List[Dict[str, Any]]:
        """Load examples from JSONL file."""
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line.strip())
                examples.append(example)
        return examples

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a single example using the prompt template."""
        instruction = example.get("prompt", example.get("instruction", ""))
        response = example.get("response", example.get("output", ""))

        return self.prompt_template.format(instruction=instruction, response=response)

    def _tokenize_and_concatenate(self, examples: List[Dict[str, Any]]) -> List[int]:
        """Tokenize all examples and concatenate them."""
        all_tokens = []

        for example in examples:
            formatted_text = self._format_example(example)

            # Tokenize with BOS token
            tokens = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            # Add EOS token at the end of each example
            if (
                self.tokenizer.eos_token_id is not None
                and tokens[-1] != self.tokenizer.eos_token_id
            ):
                tokens.append(self.tokenizer.eos_token_id)

            all_tokens.extend(tokens)

        return all_tokens

    def _pack_sequences(self, all_tokens: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Pack tokens into sequences of seq_length."""
        sequences = []

        for i in range(0, len(all_tokens) - self.seq_length, self.seq_length):
            input_ids = all_tokens[i : i + self.seq_length]
            labels = all_tokens[i + 1 : i + self.seq_length + 1]

            if len(labels) == self.seq_length:
                sequences.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

        return sequences

    def __len__(self) -> int:
        """Returns the number of sequences in this Dataset."""
        return len(self.sequences)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Returns the ith element of the Dataset.

        Args:
            i: Index, must be less than len(self)

        Returns:
            Dictionary with keys:
            - input_ids: PyTorch tensor of shape (seq_length,) with input token IDs
            - labels: PyTorch tensor of shape (seq_length,) with label token IDs
        """
        return self.sequences[i]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
    prompt_template_path: str | os.PathLike | None = None,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.
        prompt_template_path: str | None
            Path to prompt template file.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    if prompt_template_path is None:
        prompt_template_path = "cs336_alignment/prompts/alpaca_sft.prompt"

    return PackedSFTDataset(
        tokenizer, dataset_path, seq_length, shuffle, prompt_template_path
    )


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


def parse_mmlu_response(
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    patterns = [
        r"\b([ABCD])\)",  # A), B), etc.
        r"\(([ABCD])\)",  # (A), (B), etc.
        r"\b([ABCD])\.",  # A., B., etc.
        r"\b([ABCD]):",  # A:, B:, etc.
        r"[Aa]nswer\s*:?\s*([ABCD])",  # Answer: A, answer A, etc.
        r"[Oo]ption\s*:?\s*([ABCD])",  # Option: A, option A, etc.
        r"\b([ABCD])\b",  # Just the letter by itself
    ]

    for pattern in patterns:
        matches = re.findall(pattern, model_output, re.IGNORECASE)
        if not matches:
            continue

        return matches[-1].upper()

    return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    if numbers:
        return numbers[-1]
    return None


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # Format the chosen and rejected sequences using Alpaca template
    template_path = "cs336_alignment/prompts/alpaca_sft.prompt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    chosen_text = template.format(instruction=prompt, response=response_chosen)
    rejected_text = template.format(instruction=prompt, response=response_rejected)

    # Add EOS token at the end of each response
    if tokenizer.eos_token is not None:
        chosen_text += tokenizer.eos_token
        rejected_text += tokenizer.eos_token

    device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    # Tokenize sequences
    chosen = tokenizer(chosen_text, return_tensors="pt").to(device)
    rejected = tokenizer(rejected_text, return_tensors="pt").to(device)

    chosen_input_ids, chosen_labels = (
        chosen["input_ids"][:, :-1],
        chosen["input_ids"][:, 1:],
    )
    rejected_input_ids, rejected_labels = (
        rejected["input_ids"][:, :-1],
        rejected["input_ids"][:, 1:],
    )

    # Set model to evaluation mode
    lm_ref.eval()

    # Current model log probabilities
    chosen_log_prob = get_response_log_probs(lm, chosen_input_ids, chosen_labels)[
        "log_probs"
    ].sum()
    rejected_log_prob = get_response_log_probs(lm, rejected_input_ids, rejected_labels)[
        "log_probs"
    ].sum()

    # Reference model log probabilities
    with torch.no_grad():
        chosen_log_prob_ref = (
            get_response_log_probs(
                lm_ref, chosen_input_ids.to(ref_device), chosen_labels.to(ref_device)
            )["log_probs"]
            .sum()
            .to(device)
        )
        rejected_log_prob_ref = (
            get_response_log_probs(
                lm_ref,
                rejected_input_ids.to(ref_device),
                rejected_labels.to(ref_device),
            )["log_probs"]
            .sum()
            .to(device)
        )

    # Compute the DPO loss
    logits_diff = beta * (
        (chosen_log_prob - rejected_log_prob)
        - (chosen_log_prob_ref - rejected_log_prob_ref)
    )
    dpo_loss = -torch.nn.functional.logsigmoid(logits_diff)

    return dpo_loss
