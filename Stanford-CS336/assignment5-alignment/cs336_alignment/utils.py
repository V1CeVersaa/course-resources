"""Utility functions for evaluation and logging."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# ─────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────
PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt_template(name: str) -> str:
    with open(PROMPT_DIR / f"{name}.prompt") as f:
        return f.read()


def format_r1_zero_prompt(question: str) -> str:
    template = load_prompt_template("r1_zero")
    return template.format(question=question)


def format_question_only_prompt(question: str) -> str:
    template = load_prompt_template("question_only")
    return template.format(question=question)


# ─────────────────────────────────────────────
# vLLM helpers
# ─────────────────────────────────────────────
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start the inference process using vLLM on a separate GPU."""
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm):
    """Load policy weights into a vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ─────────────────────────────────────────────
# evaluate_vllm
# ─────────────────────────────────────────────
def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params,
    output_path: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(generated_text, ground_truths[i])

        results.append({
            "prompt": prompts[i],
            "ground_truth": ground_truths[i],
            "generated_text": generated_text,
            "reward": reward_dict["reward"],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
        })

        total_reward += reward_dict["reward"]
        total_format_reward += reward_dict["format_reward"]
        total_answer_reward += reward_dict["answer_reward"]

    n = len(prompts)
    metrics = {
        "mean_reward": total_reward / n,
        "mean_format_reward": total_format_reward / n,
        "mean_answer_reward": total_answer_reward / n,
        "num_examples": n,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics


# ─────────────────────────────────────────────
# log_generations
# ─────────────────────────────────────────────
def log_generations(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params,
    policy_model: torch.nn.Module | None = None,
    tokenizer=None,
    step: int = 0,
    prefix: str = "eval",
    wandb_log: bool = True,
):
    """Log generations from a model, including rewards and entropy."""
    from cs336_alignment.alignment import (
        compute_entropy,
        get_response_log_probs,
        tokenize_prompt_and_output,
    )

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    log_data = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_response_len = 0
    correct_response_lens = []
    incorrect_response_lens = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        reward_dict = reward_fn(generated_text, ground_truths[i])

        response_len = len(generated_text.split())
        total_response_len += response_len
        total_reward += reward_dict["reward"]
        total_format_reward += reward_dict["format_reward"]
        total_answer_reward += reward_dict["answer_reward"]

        if reward_dict["answer_reward"] > 0:
            correct_response_lens.append(response_len)
        else:
            incorrect_response_lens.append(response_len)

        entry = {
            "prompt": prompts[i],
            "generated_text": generated_text,
            "ground_truth": ground_truths[i],
            "reward": reward_dict["reward"],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
            "response_length": response_len,
        }

        # Compute per-token entropy if we have the policy model
        if policy_model is not None and tokenizer is not None:
            tok_out = tokenize_prompt_and_output(
                [prompts[i]], [generated_text], tokenizer
            )
            device = next(policy_model.parameters()).device
            input_ids = tok_out["input_ids"].to(device)
            labels = tok_out["labels"].to(device)
            with torch.no_grad():
                lp_out = get_response_log_probs(
                    policy_model, input_ids, labels, return_token_entropy=True
                )
            mask = tok_out["response_mask"].to(device).float()
            if "token_entropy" in lp_out:
                avg_entropy = (lp_out["token_entropy"] * mask).sum() / mask.sum().clamp(min=1)
                entry["avg_token_entropy"] = avg_entropy.item()

        log_data.append(entry)

    n = len(prompts)
    metrics = {
        f"{prefix}/mean_reward": total_reward / n,
        f"{prefix}/mean_format_reward": total_format_reward / n,
        f"{prefix}/mean_answer_reward": total_answer_reward / n,
        f"{prefix}/mean_response_length": total_response_len / n,
    }
    if correct_response_lens:
        metrics[f"{prefix}/mean_correct_response_length"] = (
            sum(correct_response_lens) / len(correct_response_lens)
        )
    if incorrect_response_lens:
        metrics[f"{prefix}/mean_incorrect_response_length"] = (
            sum(incorrect_response_lens) / len(incorrect_response_lens)
        )

    if wandb_log:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

    # Print a few examples
    for entry in log_data[:5]:
        print(f"--- Prompt: {entry['prompt'][:100]}...")
        print(f"    Response: {entry['generated_text'][:200]}...")
        print(f"    Ground truth: {entry['ground_truth']}")
        print(f"    Reward: {entry['reward']:.2f} "
              f"(format: {entry['format_reward']:.2f}, "
              f"answer: {entry['answer_reward']:.2f})")
        if "avg_token_entropy" in entry:
            print(f"    Avg entropy: {entry['avg_token_entropy']:.4f}")
        print()

    return metrics, log_data
