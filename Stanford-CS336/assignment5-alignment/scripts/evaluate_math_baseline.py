"""
Evaluate zero-shot MATH performance of Qwen 2.5 Math 1.5B Base.

Problem (math_baseline): 4 points
"""
import json
import typer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import evaluate_vllm, format_r1_zero_prompt, init_vllm


def main(
    model_id: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    validation_path: str = "/data/a5-alignment/MATH/validation.jsonl",
    output_path: str = "outputs/math_baseline_results.json",
    device: str = "cuda:0",
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
):
    from vllm import SamplingParams

    # Load validation data
    examples = []
    with open(validation_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Format prompts using r1_zero
    prompts = [format_r1_zero_prompt(ex["problem"]) for ex in examples]
    ground_truths = [ex["answer"] for ex in examples]

    # Set up sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initialize vLLM
    llm = init_vllm(model_id, device=device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    # Evaluate
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=output_path,
    )

    print("=== Zero-shot MATH Baseline Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    typer.run(main)
