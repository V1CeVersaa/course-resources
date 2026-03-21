"""Core alignment implementations for CS336 Assignment 5."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# ─────────────────────────────────────────────
# tokenize_prompt_and_output
# ─────────────────────────────────────────────
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt+output, build input_ids/labels/response_mask."""
    batch_size = len(prompt_strs)

    # Tokenize prompts and outputs separately (no special tokens added)
    prompt_encodings = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_encodings = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    # Concatenate prompt + output for each example
    full_encodings = [p + o for p, o in zip(prompt_encodings, output_encodings)]
    prompt_lens = [len(p) for p in prompt_encodings]
    full_lens = [len(f) for f in full_encodings]
    max_len = max(full_lens)

    # Pad token id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Build padded full sequences of length max_len
    padded = []
    for enc in full_encodings:
        padded.append(enc + [pad_id] * (max_len - len(enc)))

    padded_tensor = torch.tensor(padded, dtype=torch.long)  # (batch_size, max_len)

    # input_ids: all tokens except the last one
    input_ids = padded_tensor[:, :-1]  # (batch_size, max_len - 1)
    # labels: all tokens except the first one (shifted)
    labels = padded_tensor[:, 1:]  # (batch_size, max_len - 1)

    # response_mask: 1 for response tokens in labels, 0 for prompt/padding
    # In labels (shifted by 1), position j corresponds to predicting token j+1
    # Response tokens in the original sequence start at prompt_len
    # In labels, the response region is [prompt_len-1, full_len-1)
    # But we want mask on labels, so position j in labels is 1 if j+1 is a response token
    # Token at original position i is a response token if i >= prompt_len and i < full_len
    # In labels, position j predicts original position j+1
    # So labels[j] is response if j+1 >= prompt_len and j+1 < full_len
    # i.e., j >= prompt_len - 1 and j < full_len - 1
    # But since input_ids has max_len-1 positions, j ranges from 0 to max_len-2
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i in range(batch_size):
        # labels position j corresponds to original token j+1
        # Response starts at original index prompt_lens[i]
        # So in labels, j+1 >= prompt_lens[i] and j+1 < full_lens[i]
        # i.e. j >= prompt_lens[i] - 1 and j < full_lens[i] - 1
        resp_start = max(prompt_lens[i] - 1, 0)
        resp_end = full_lens[i] - 1
        response_mask[i, resp_start:resp_end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


# ─────────────────────────────────────────────
# compute_entropy
# ─────────────────────────────────────────────
def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropy of logits. Shape: (B, T, V) -> (B, T)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


# ─────────────────────────────────────────────
# get_response_log_probs
# ─────────────────────────────────────────────
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Get per-token log-probs (and optionally entropy) from a model."""
    logits = model(input_ids).logits  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather log-probs for each label token
    # labels shape: (B, T)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    result = {"log_probs": token_log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


# ─────────────────────────────────────────────
# masked_normalize
# ─────────────────────────────────────────────
def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """Sum over masked elements along dim, then divide by normalize_constant."""
    mask = mask.float()
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    else:
        return masked.sum(dim=dim) / normalize_constant


# ─────────────────────────────────────────────
# masked_mean
# ─────────────────────────────────────────────
def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """Compute mean of tensor where mask==1, along given dim."""
    mask = mask.float()
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1)
    else:
        return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


# ─────────────────────────────────────────────
# sft_microbatch_train_step
# ─────────────────────────────────────────────
def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """SFT microbatch: compute NLL loss, backprop with grad accumulation scaling."""
    # NLL loss = -log_probs, masked and normalized
    nll = -policy_log_probs  # (B, T)
    loss = masked_normalize(nll, response_mask, dim=-1, normalize_constant=normalize_constant)
    # Average over batch
    loss = loss.mean()
    # Scale by gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss.detach(), {}


# ─────────────────────────────────────────────
# compute_group_normalized_rewards
# ─────────────────────────────────────────────
def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute group-normalized rewards (advantages) for GRPO."""
    n = len(rollout_responses)

    # Compute raw rewards
    raw_rewards_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        raw_rewards_list.append(reward_dict["reward"])

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # Reshape into groups: (n_groups, group_size)
    n_groups = n // group_size
    grouped = raw_rewards.view(n_groups, group_size)

    # Compute group mean and std
    group_mean = grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)

    if normalize_by_std:
        group_std = grouped.std(dim=1, keepdim=True)  # (n_groups, 1)
        normalized = (grouped - group_mean) / (group_std + advantage_eps)
    else:
        normalized = grouped - group_mean

    # Flatten back
    normalized_rewards = normalized.view(n)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return normalized_rewards, raw_rewards, metadata


# ─────────────────────────────────────────────
# compute_naive_policy_gradient_loss
# ─────────────────────────────────────────────
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """Per-token policy gradient loss: -A * log_prob."""
    # raw_rewards_or_advantages: (B, 1), policy_log_probs: (B, T)
    return -raw_rewards_or_advantages * policy_log_probs


# ─────────────────────────────────────────────
# compute_grpo_clip_loss
# ─────────────────────────────────────────────
def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute GRPO-Clip per-token loss (Eq. 33 in the PDF)."""
    # Compute importance sampling ratio
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # Per-token loss: -min(ratio * A, clipped_ratio * A)
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    loss = -torch.min(loss1, loss2)

    # Metadata: track clipping
    clipped = (ratio != clipped_ratio).float()
    metadata = {"clip_fraction": clipped}

    return loss, metadata


# ─────────────────────────────────────────────
# compute_policy_gradient_loss (wrapper)
# ─────────────────────────────────────────────
def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatch to the appropriate policy gradient loss."""
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_no_clip":
        # Eq. 34: unclipped off-policy loss -(π_θ/π_θ_old) * A_t
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        loss = -(ratio * advantages)
        return loss, {}
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ─────────────────────────────────────────────
# grpo_microbatch_train_step
# ─────────────────────────────────────────────
def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """GRPO microbatch: compute PG loss, mask, average, backprop."""
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Masked mean over sequence dim, then mean over batch
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=-1)  # (B,)
    loss = per_example_loss.mean()

    # Scale for gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    return loss.detach(), metadata


# ─────────────────────────────────────────────
# Packed SFT Dataset
# ─────────────────────────────────────────────
class PackedSFTDataset(Dataset):
    """Dataset that packs instruction-tuning examples to constant length."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        # Load and tokenize all examples
        examples = []
        with open(dataset_path) as f:
            for line in f:
                examples.append(json.loads(line))

        if shuffle:
            import random
            random.shuffle(examples)

        # Tokenize and concatenate all examples with BOS/EOS
        all_tokens = []
        for ex in examples:
            prompt = ex.get("prompt", ex.get("instruction", ""))
            response = ex.get("response", ex.get("output", ""))
            text = prompt + response
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)

        # Pack into sequences of seq_length + 1 (for input/label shift)
        self.examples = []
        for i in range(0, len(all_tokens) - seq_length, seq_length):
            chunk = all_tokens[i : i + seq_length + 1]
            if len(chunk) == seq_length + 1:
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


# ─────────────────────────────────────────────
# iterate_batches
# ─────────────────────────────────────────────
def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────
# parse_mmlu_response
# ─────────────────────────────────────────────
def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse MMLU model output to extract predicted option letter (A/B/C/D)."""
    valid_letters = {"A", "B", "C", "D"}

    # Try to find patterns like "The correct answer is X" or "answer is X"
    patterns = [
        r'\b(?:answer|option)\s+(?:is\s+)?([A-D])\b',
        r'\b([A-D])\s*[\.\):]',
        r'^([A-D])$',
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    return None


# ─────────────────────────────────────────────
# parse_gsm8k_response
# ─────────────────────────────────────────────
def parse_gsm8k_response(model_output: str) -> str | None:
    """Extract the last number from GSM8K model output."""
    # Find all numbers (including negative, decimals, commas)
    numbers = re.findall(r'-?[\d,]+\.?\d*', model_output)
    if numbers:
        # Return the last number, removing commas
        return numbers[-1].replace(",", "")
    return None


# ─────────────────────────────────────────────
# compute_per_instance_dpo_loss
# ─────────────────────────────────────────────
def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """Compute DPO loss for a single preference pair."""

    def get_log_probs_sum(model, prompt_str, response_str):
        """Get sum of log-probs of response tokens given prompt."""
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        response_ids = tokenizer.encode(response_str, add_special_tokens=False)
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
        labels = torch.tensor([full_ids[1:]], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Only sum over response tokens
        # In the shifted setup, response tokens in labels start at index len(prompt_ids)-1
        prompt_len = len(prompt_ids)
        response_log_probs = token_log_probs[0, prompt_len - 1:]
        return response_log_probs.sum()

    # Get log-probs for chosen and rejected under both models
    log_prob_chosen = get_log_probs_sum(lm, prompt, response_chosen)
    log_prob_rejected = get_log_probs_sum(lm, prompt, response_rejected)
    ref_log_prob_chosen = get_log_probs_sum(lm_ref, prompt, response_chosen)
    ref_log_prob_rejected = get_log_probs_sum(lm_ref, prompt, response_rejected)

    # DPO loss: -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    log_ratio_chosen = log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = log_prob_rejected - ref_log_prob_rejected

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss
