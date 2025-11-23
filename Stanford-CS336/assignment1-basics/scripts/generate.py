from pathlib import Path

import torch
import yaml
from config_setting import GenerationConfig, ModelConfig
from loguru import logger
from pydantic import BaseModel

from cs336_basics import Tokenizer, TransformerLM, load_checkpoint, softmax


class Config(BaseModel):
    model: ModelConfig
    generation: GenerationConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    return softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0 or p <= 0.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens with cumulative probability <= p (include first token where cumulative >= p)
    cutoff = torch.searchsorted(cumulative_probs, torch.tensor(p, device=probs.device))
    cutoff = int(cutoff.clamp(min=0, max=probs.numel() - 1))

    # Mask out tokens beyond cutoff in the original probability vector
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[sorted_indices[cutoff + 1 :]] = True
    probs = probs.masked_fill(mask, 0.0)

    total = probs.sum()
    if total.item() == 0.0:
        # Fallback to the highest probability token if everything got masked
        probs.zero_()
        probs[sorted_indices[0]] = 1.0
        return probs

    return probs / total


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> str:
    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    input_ids = tokenizer.encode(prompt)
    eos_id = None
    eos_bytes = b"<|endoftext|>"
    if eos_bytes in tokenizer.token_to_id:
        eos_id = tokenizer.token_to_id[eos_bytes]

    model.eval()
    generated = list(input_ids)

    with torch.no_grad():
        while len(generated) < max_length:
            # Respect context length
            context = generated[-model.context_length :]
            x = torch.tensor([context], dtype=torch.long, device=device)

            logits = model(x)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1]  # (vocab_size,)

            probs = softmax_with_temperature(next_logits, temperature)

            if top_p < 1.0:
                probs = top_p_filter(probs, top_p)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(int(next_token))

            if eos_id is not None and next_token == eos_id:
                break

    return tokenizer.decode(generated)


def main():
    config_path = Path(__file__).parent / f"config/conf_{Path(__file__).stem}.yaml"
    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    device = torch.device(cfg.generation.device)
    logger.info(f"Using device: {device}")

    model = TransformerLM(**cfg.model.model_dump())
    model.to(device)

    load_checkpoint(cfg.generation.checkpoint_path, model)

    tokenizer = Tokenizer.from_file(
        cfg.generation.tokenizer_vocab, cfg.generation.tokenizer_merges, cfg.generation.tokenizer_specs
    )

    logger.info(f"Temperature: {cfg.generation.temperature}")
    logger.info(f"Max Length: {cfg.generation.max_length}")
    logger.info(f"Initial Prompts: {cfg.generation.prompt}")

    out = generate_text(
        model,
        tokenizer,
        cfg.generation.prompt,
        max_length=cfg.generation.max_length,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        seed=cfg.generation.seed,
    )

    logger.info(f"Generated Text: {out}")


if __name__ == "__main__":
    main()
