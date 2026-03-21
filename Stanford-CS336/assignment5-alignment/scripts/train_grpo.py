"""
GRPO training script for MATH dataset.

Problem (grpo_train_loop): 5 points
Algorithm 3 from the assignment.
"""
from __future__ import annotations

import json
import os
from typing import Literal

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from cs336_alignment.alignment import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_mean,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    evaluate_vllm,
    format_r1_zero_prompt,
    init_vllm,
    load_policy_into_vllm_instance,
)


def main(
    model_id: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    train_data_path: str = "/data/a5-alignment/MATH/train.jsonl",
    validation_path: str = "/data/a5-alignment/MATH/validation.jsonl",
    output_dir: str = "outputs/grpo",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    eval_every: int = 5,
    eval_size: int = 1024,
    seed: int = 42,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    wandb_project: str = "cs336-a5-grpo",
    prompt_type: str = "r1_zero",
):
    from vllm import SamplingParams

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(project=wandb_project, config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load train data
    train_examples = []
    with open(train_data_path) as f:
        for line in f:
            train_examples.append(json.loads(line))

    # Load validation data
    val_examples = []
    with open(validation_path) as f:
        for line in f:
            val_examples.append(json.loads(line))

    # Choose prompt formatter and reward function based on prompt_type
    if prompt_type == "r1_zero":
        format_prompt = format_r1_zero_prompt
        reward_fn = r1_zero_reward_fn
    elif prompt_type == "question_only":
        from cs336_alignment.drgrpo_grader import question_only_reward_fn
        from cs336_alignment.utils import format_question_only_prompt
        format_prompt = format_question_only_prompt
        reward_fn = question_only_reward_fn
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    eval_prompts = [format_r1_zero_prompt(ex["problem"]) for ex in val_examples[:eval_size]]
    eval_ground_truths = [ex["answer"] for ex in val_examples[:eval_size]]

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initialize vLLM
    llm = init_vllm(model_id, device=vllm_device, seed=seed,
                     gpu_memory_utilization=gpu_memory_utilization)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=0.0, betas=(0.9, 0.95),
    )

    import random
    random.seed(seed)
    global_step = 0

    for grpo_step in tqdm(range(n_grpo_steps), desc="GRPO"):
        # ── Step 1: Sample a batch of questions ──
        batch_questions = random.choices(train_examples, k=n_prompts_per_rollout_batch)
        prompts = [format_prompt(ex["problem"]) for ex in batch_questions]
        ground_truths = [ex["answer"] for ex in batch_questions]

        # ── Step 2: Set old policy = current policy, generate rollouts ──
        model.eval()
        load_policy_into_vllm_instance(model, llm)

        outputs = llm.generate(prompts, rollout_sampling_params)

        # Flatten: each prompt produces group_size outputs
        rollout_prompts = []
        rollout_responses = []
        repeated_ground_truths = []
        for i, output in enumerate(outputs):
            for completion in output.outputs:
                rollout_prompts.append(prompts[i])
                rollout_responses.append(completion.text)
                repeated_ground_truths.append(ground_truths[i])

        # ── Step 3: Compute rewards and advantages ──
        normalized_rewards, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        wandb.log({
            "train/mean_reward": raw_rewards.mean().item(),
            "train/mean_answer_reward": reward_metadata.get("mean_reward", 0),
            "train_step": global_step,
        })

        # ── Step 4: Tokenize all rollouts ──
        tok_out = tokenize_prompt_and_output(
            rollout_prompts, rollout_responses, tokenizer
        )

        all_input_ids = tok_out["input_ids"].to(policy_device)
        all_labels = tok_out["labels"].to(policy_device)
        all_response_mask = tok_out["response_mask"].to(policy_device)

        # Compute old log probs (for off-policy / GRPO-Clip)
        old_log_probs_all = None
        if loss_type == "grpo_clip" or epochs_per_rollout_batch > 1:
            with torch.no_grad():
                old_lp_out = get_response_log_probs(
                    model, all_input_ids, all_labels, return_token_entropy=False
                )
            old_log_probs_all = old_lp_out["log_probs"].detach()

        # ── Step 5: Train on the rollout batch ──
        model.train()
        for epoch in range(epochs_per_rollout_batch):
            # Shuffle microbatch indices
            indices = torch.randperm(rollout_batch_size)

            for mb_start in range(0, rollout_batch_size, micro_train_batch_size):
                mb_idx = indices[mb_start : mb_start + micro_train_batch_size]

                mb_input_ids = all_input_ids[mb_idx]
                mb_labels = all_labels[mb_idx]
                mb_response_mask = all_response_mask[mb_idx]
                mb_advantages = normalized_rewards[mb_idx].unsqueeze(1).to(policy_device)
                mb_raw_rewards = raw_rewards[mb_idx].unsqueeze(1).to(policy_device)

                # Forward pass
                lp_out = get_response_log_probs(
                    model, mb_input_ids, mb_labels, return_token_entropy=True
                )
                policy_log_probs = lp_out["log_probs"]

                mb_old_log_probs = None
                if old_log_probs_all is not None:
                    mb_old_log_probs = old_log_probs_all[mb_idx]

                # Determine if this is the start of a new optimizer step
                mb_position = (mb_start // micro_train_batch_size) % gradient_accumulation_steps
                if mb_position == 0:
                    optimizer.zero_grad()

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )

                # Step optimizer every gradient_accumulation_steps microbatches
                if (mb_position + 1) == gradient_accumulation_steps or \
                   mb_start + micro_train_batch_size >= rollout_batch_size:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    global_step += 1

                    # Log entropy
                    if "token_entropy" in lp_out:
                        avg_entropy = masked_mean(
                            lp_out["token_entropy"], mb_response_mask
                        ).item()
                    else:
                        avg_entropy = 0.0

                    grad_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5

                    log_dict = {
                        "train/loss": loss.item(),
                        "train/entropy": avg_entropy,
                        "train/grad_norm": grad_norm,
                        "train_step": global_step,
                    }
                    if "clip_fraction" in metadata:
                        clip_frac = masked_mean(
                            metadata["clip_fraction"], mb_response_mask
                        ).item()
                        log_dict["train/clip_fraction"] = clip_frac

                    wandb.log(log_dict)

        # ── Step 6: Evaluate ──
        if (grpo_step + 1) % eval_every == 0 or grpo_step == 0:
            model.eval()
            load_policy_into_vllm_instance(model, llm)

            metrics = evaluate_vllm(
                vllm_model=llm,
                reward_fn=r1_zero_reward_fn,
                prompts=eval_prompts,
                ground_truths=eval_ground_truths,
                eval_sampling_params=eval_sampling_params,
            )

            wandb.log({
                "eval/mean_reward": metrics["mean_reward"],
                "eval/mean_format_reward": metrics["mean_format_reward"],
                "eval/mean_answer_reward": metrics["mean_answer_reward"],
                "eval_step": global_step,
            })
            print(f"\nStep {grpo_step + 1}: "
                  f"answer_reward={metrics['mean_answer_reward']:.4f}")

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
