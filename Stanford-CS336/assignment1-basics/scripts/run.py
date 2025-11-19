import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from cs336_basics import (
    AdamW,
    TransformerLM,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-data", type=str, required=True, help="Path to training data (numpy memmap file)")
    data_group.add_argument("--val-data", type=str, required=True, help="Path to validation data (numpy memmap file)")
    data_group.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int64"],
        help="Data type of memmap files",
    )

    # Model hyperparameters
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    model_group.add_argument("--context-length", type=int, default=256, help="Maximum context length")
    model_group.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    model_group.add_argument("--d-model", type=int, default=512, help="Model dimension")
    model_group.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    model_group.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    model_group.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Optimizer hyperparameters
    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument("--learning-rate", type=float, default=3e-4, help="Maximum learning rate")
    optim_group.add_argument("--min-learning-rate", type=float, default=3e-5, help="Minimum learning rate")
    optim_group.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    optim_group.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    optim_group.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    optim_group.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    optim_group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max L2 norm")
    optim_group.add_argument("--warmup-iters", type=int, default=1000, help="Number of warmup iterations")

    # Training hyperparameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_group.add_argument("--max-iters", type=int, default=100000, help="Maximum number of training iterations")
    train_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    train_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Logging & Checkpointing
    log_group = parser.add_argument_group("Logging & Checkpointing")
    log_group.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    log_group.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    log_group.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N iterations")
    log_group.add_argument("--eval-every", type=int, default=500, help="Evaluate on validation set every N iterations")
    log_group.add_argument("--eval-batches", type=int, default=20, help="Number of batches for validation evaluation")
    log_group.add_argument("--log-every", type=int, default=100, help="Log training metrics every N iterations")

    # Weights & Biases
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    wandb_group.add_argument("--wandb-project", type=str, default="cs336-transformer-lm", help="W&B project name")
    wandb_group.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    return parser.parse_args()


def load_memmap_data(data_path: str, dtype_str: str = "uint16"):
    """Load data using memory-mapped array for memory-efficient access."""
    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int64": np.int64,
    }
    dtype = dtype_map[dtype_str]

    logger.info(f"Loading data from {data_path} (dtype={dtype_str})")
    data = np.memmap(data_path, dtype=dtype, mode="r")
    logger.info(f"  ✓ Loaded {len(data):,} tokens")
    return data


@torch.no_grad()
def evaluate_model(
    model: TransformerLM, val_data: np.ndarray, batch_size: int, context_length: int, device: str, num_batches: int = 20
) -> float:
    """Evaluate model on validation set and return average loss."""
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)

        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        loss = cross_entropy(logits_flat, y_flat)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def train(args):
    """Main training function."""

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Initialize Weights & Biases
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            logger.info(f"W&B initialized: {args.wandb_project}")
        except ImportError:
            logger.warning("wandb not available, disabling W&B logging")
            use_wandb = False

    # ==================== Data ====================
    train_data = load_memmap_data(args.train_data, args.data_dtype)
    val_data = load_memmap_data(args.val_data, args.data_dtype)

    # ==================== Model ====================
    logger.info("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model has {num_params:,} parameters")
    logger.info(f"  ✓ Device: {args.device}")

    # ==================== Optimizer ====================
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    logger.info("  ✓ Optimizer initialized")

    # ==================== Resume from Checkpoint ====================
    start_iter = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"  ✓ Resumed from iteration {start_iter}")

    # ==================== Training Loop ====================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting training from iteration {start_iter}")
    logger.info(f"{'=' * 60}\n")

    model.train()

    for iter_num in range(start_iter, args.max_iters):
        # Update learning rate with cosine schedule
        lr = get_lr_cosine_schedule(
            it=iter_num,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get training batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass
        logits = model(x)

        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        loss = cross_entropy(logits_flat, y_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # ==================== Logging ====================
        if (iter_num + 1) % args.log_every == 0:
            progress = (iter_num + 1) / args.max_iters * 100
            logger.info(
                f"[{progress:5.1f}%] Iter {iter_num + 1:6d}/{args.max_iters} | Loss: {loss.item():.4f} | LR: {lr:.6f}"
            )

            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                    "iteration": iter_num + 1,
                })

        # ==================== Validation ====================
        if (iter_num + 1) % args.eval_every == 0:
            logger.info("Running validation...")
            val_loss = evaluate_model(
                model, val_data, args.batch_size, args.context_length, args.device, args.eval_batches
            )
            logger.info(f"  ✓ Validation loss: {val_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "iteration": iter_num + 1,
                })

        # ==================== Checkpointing ====================
        if (iter_num + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num + 1}.pt"
            save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
            logger.info(f"  ✓ Checkpoint saved: {checkpoint_path.name}")

    # ==================== Final Checkpoint ====================
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)

    logger.info(f"\n{'=' * 60}")
    logger.info("Training complete!")
    logger.info(f"Final checkpoint: {final_checkpoint_path}")
    logger.info(f"{'=' * 60}\n")

    if use_wandb:
        wandb.finish()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
