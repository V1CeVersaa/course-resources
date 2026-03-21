"""
SFT training script for MATH dataset.

Problem (sft_experiment): 2 points
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from cs336_alignment.alignment import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    evaluate_vllm,
    format_r1_zero_prompt,
    init_vllm,
    load_policy_into_vllm_instance,
    log_generations,
)


def main(
    model_id: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    sft_data_path: str = "/data/a5-alignment/MATH/sft.jsonl",
    validation_path: str = "/data/a5-alignment/MATH/validation.jsonl",
    output_dir: str = "outputs/sft",
    n_sft_steps: int = 1000,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-5,
    max_num_examples: int | None = None,
    eval_every: int = 50,
    eval_size: int = 1024,
    seed: int = 42,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    gpu_memory_utilization: float = 0.85,
    wandb_project: str = "cs336-a5-sft",
):
    from vllm import SamplingParams

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
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

    # Load SFT data
    sft_examples = []
    with open(sft_data_path) as f:
        for line in f:
            sft_examples.append(json.loads(line))
    if max_num_examples is not None:
        sft_examples = sft_examples[:max_num_examples]
    print(f"Loaded {len(sft_examples)} SFT examples")

    # Load validation data
    val_examples = []
    with open(validation_path) as f:
        for line in f:
            val_examples.append(json.loads(line))

    eval_prompts = [format_r1_zero_prompt(ex["problem"]) for ex in val_examples[:eval_size]]
    eval_ground_truths = [ex["answer"] for ex in val_examples[:eval_size]]

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    # Initialize vLLM for evaluation
    llm = init_vllm(model_id, device=vllm_device, seed=seed,
                     gpu_memory_utilization=gpu_memory_utilization)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    # Training loop
    model.train()
    global_step = 0
    data_idx = 0

    for step in tqdm(range(n_sft_steps), desc="SFT Training"):
        optimizer.zero_grad()

        for micro_step in range(gradient_accumulation_steps):
            # Sample a batch
            batch_prompts = []
            batch_outputs = []
            for _ in range(batch_size):
                ex = sft_examples[data_idx % len(sft_examples)]
                batch_prompts.append(ex["prompt"])
                batch_outputs.append(ex["response"])
                data_idx += 1

            # Tokenize
            tok_out = tokenize_prompt_and_output(batch_prompts, batch_outputs, tokenizer)
            input_ids = tok_out["input_ids"].to(policy_device)
            labels = tok_out["labels"].to(policy_device)
            response_mask = tok_out["response_mask"].to(policy_device)

            # Forward pass
            lp_out = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            policy_log_probs = lp_out["log_probs"]

            # Microbatch train step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1

        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train_step": global_step,
        })

        # Evaluate periodically
        if global_step % eval_every == 0 or global_step == 1:
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

            print(f"Step {global_step}: answer_reward={metrics['mean_answer_reward']:.4f}")
            model.train()

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
