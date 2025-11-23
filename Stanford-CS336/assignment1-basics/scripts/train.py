from pathlib import Path

import torch
import wandb
import yaml
from config_setting import CheckLogConfig, DatasetConfig, ModelConfig, OptimizerConfig
from loguru import logger
from pydantic import BaseModel

from cs336_basics import (
    AdamW,
    Dataset,
    TransformerLM,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
)


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    checklog: CheckLogConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def evaluate(model, dataset, cfg, optimizer, iter_num, lr):
    model.eval()
    total_loss = 0.0

    for _ in range(cfg.checklog.eval_iters):
        data, label = dataset.get_batch("valid")

        with torch.no_grad():
            logits = model(data)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1))
            total_loss += loss.item()

    total_loss /= cfg.checklog.eval_iters
    logger.info(f"Iter: {iter_num}, Val loss: {total_loss:.4f}, LR: {lr:.6f}")

    if cfg.checklog.wandb_logging:
        wandb.log({"val_loss": total_loss, "lr": lr, "iter": iter_num})

    ckpt_path = cfg.checklog.checkpoint_dir / f"{cfg.checklog.wandb_run_name}_step_{iter_num}.pt"
    save_checkpoint(model, optimizer, iter_num, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    model.train()


def main():
    logger.info(f"load config from config/conf_{Path(__file__).stem}.yaml")
    cfg = load_config(f"config/conf_{Path(__file__).stem}.yaml")

    if cfg.checklog.wandb_logging:
        wandb.init(
            project=cfg.checklog.wandb_project,
            name=cfg.checklog.wandb_run_name,
            config=cfg.model_dump(),
        )

    dataset = Dataset(**cfg.dataset.model_dump())
    model = TransformerLM(**cfg.model.model_dump())
    model.to(cfg.dataset.device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        weight_decay=cfg.optimizer.weight_decay,
    )

    for iter in range(cfg.optimizer.total_iters):
        optimizer.zero_grad()
        data, label = dataset.get_batch("train")

        logits = model(data)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1))
        loss.backward()

        gradient_clipping(model.parameters(), 1.0)
        lr = get_lr_cosine_schedule(
            iter,
            cfg.optimizer.lr_max,
            cfg.optimizer.lr_min,
            cfg.optimizer.warmup_iters,
            cfg.optimizer.total_iters,
        )
        optimizer.set_lr(lr)
        optimizer.step()

        if iter % cfg.checklog.log_interval == 0:
            logger.info(f"Iter: {iter}, Train loss: {loss.item():.4f}, Learning Rate: {lr:.6f}")
            if cfg.checklog.wandb_logging:
                wandb.log({"train_loss": loss.item(), "lr": lr, "iter": iter})

        if iter % cfg.checklog.eval_interval == 0:
            evaluate(model, dataset, cfg, optimizer, iter, lr)


if __name__ == "__main__":
    main()
