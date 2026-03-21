"""
Expert Iteration training script for MATH dataset.

Problem (expert_iteration_experiment): 2 points
Algorithm 2 from the assignment.
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
)


def main(
    model_id: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    train_data_path: str = "/data/a5-alignment/MATH/train.jsonl",
    validation_path: str = "/data/a5-alignment/MATH/validation.jsonl",
    output_dir: str = "outputs/expert_iteration",
    n_ei_steps: int = 5,
    group_size: int = 8,
    batch_size: int = 512,
    sft_epochs: int = 2,
    sft_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-5,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 1024,
    sampling_min_tokens: int = 4,
    eval_every: int = 50,
    eval_size: int = 1024,
    seed: int = 42,
    policy_device: str = "cuda:0",
    vllm_device: str = "cuda:1",
    gpu_memory_utilization: float = 0.85,
    wandb_project: str = "cs336-a5-ei",
):
    from vllm import SamplingParams

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    global_step = 0

    for ei_step in range(n_ei_steps):
        print(f"\n=== Expert Iteration Step {ei_step + 1}/{n_ei_steps} ===")

        # Set old policy = current policy
        model.eval()
        load_policy_into_vllm_instance(model, llm)

        # Sample a batch of questions
        import random
        random.seed(seed + ei_step)
        batch_questions = random.sample(train_examples, min(batch_size, len(train_examples)))

        # Generate rollouts for each question
        prompts = [format_r1_zero_prompt(ex["problem"]) for ex in batch_questions]
        ground_truths = [ex["answer"] for ex in batch_questions]

        outputs = llm.generate(prompts, rollout_sampling_params)

        # Filter for correct responses -> SFT dataset
        sft_data = []
        total_generated = 0
        total_correct = 0
        for i, output in enumerate(outputs):
            for completion in output.outputs:
                total_generated += 1
                response_text = completion.text
                reward = r1_zero_reward_fn(response_text, ground_truths[i])
                if reward["answer_reward"] > 0:
                    total_correct += 1
                    sft_data.append({
                        "prompt": prompts[i],
                        "response": response_text,
                    })

        print(f"  Generated {total_generated} rollouts, {total_correct} correct "
              f"({total_correct / total_generated * 100:.1f}%)")
        print(f"  SFT dataset size: {len(sft_data)}")

        if len(sft_data) == 0:
            print("  No correct rollouts, skipping SFT step")
            continue

        # SFT on the filtered dataset
        model.train()
        for epoch in range(sft_epochs):
            random.shuffle(sft_data)
            data_idx = 0

            n_sft_batches = len(sft_data) // (sft_batch_size * gradient_accumulation_steps)
            for sft_step in range(max(n_sft_batches, 1)):
                optimizer.zero_grad()

                for micro_step in range(gradient_accumulation_steps):
                    batch_prompts = []
                    batch_outputs = []
                    for _ in range(sft_batch_size):
                        ex = sft_data[data_idx % len(sft_data)]
                        batch_prompts.append(ex["prompt"])
                        batch_outputs.append(ex["response"])
                        data_idx += 1

                    tok_out = tokenize_prompt_and_output(
                        batch_prompts, batch_outputs, tokenizer
                    )
                    input_ids = tok_out["input_ids"].to(policy_device)
                    labels = tok_out["labels"].to(policy_device)
                    response_mask = tok_out["response_mask"].to(policy_device)

                    lp_out = get_response_log_probs(
                        model, input_ids, labels, return_token_entropy=False
                    )

                    loss, _ = sft_microbatch_train_step(
                        policy_log_probs=lp_out["log_probs"],
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1

                wandb.log({
                    "train/loss": loss.item(),
                    "train/ei_step": ei_step,
                    "train_step": global_step,
                })

        # Evaluate after each EI step
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
            "eval/mean_answer_reward": metrics["mean_answer_reward"],
            "eval/mean_format_reward": metrics["mean_format_reward"],
            "eval_step": global_step,
        })
        print(f"  Eval answer_reward: {metrics['mean_answer_reward']:.4f}")

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
