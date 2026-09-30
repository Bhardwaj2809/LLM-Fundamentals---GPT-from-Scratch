"""Train MiniGPT on Tiny Shakespeare and save the best checkpoint."""

import argparse
from pathlib import Path

import torch

try:  # Supports both `python src/train.py` and `python -m src.train`.
    from .config import (
        CHECKPOINT_PATH,
        DEVICE,
        EVAL_INTERVAL,
        EVAL_ITERS,
        LEARNING_RATE,
        MAX_ITERS,
        SEED,
        WEIGHT_DECAY,
        set_seed,
    )
    from .dataset import TextDataset
    from .model import GPTLanguageModel
except ImportError:
    from config import (
        CHECKPOINT_PATH,
        DEVICE,
        EVAL_INTERVAL,
        EVAL_ITERS,
        LEARNING_RATE,
        MAX_ITERS,
        SEED,
        WEIGHT_DECAY,
        set_seed,
    )
    from dataset import TextDataset
    from model import GPTLanguageModel


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel, dataset: TextDataset, eval_iters: int
) -> dict[str, float]:
    """Estimate mean next-token loss on randomly sampled train and val batches.

    Evaluation mode disables dropout. The previous training/evaluation state is
    restored afterwards, so this function is safe to call inside training.
    """
    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}

    for split in ("train", "val"):
        split_losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            inputs, targets = dataset.get_batch(split)
            _, loss = model(inputs, targets)
            assert loss is not None
            split_losses[step] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train(was_training)
    return losses


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    validation_loss: float,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> None:
    """Save everything needed to resume training or generate text later."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


def train(max_iters: int = MAX_ITERS, eval_iters: int = EVAL_ITERS) -> None:
    """Run the MiniGPT optimization loop with AdamW and validation checkpoints."""
    set_seed(SEED)
    dataset = TextDataset()
    model = GPTLanguageModel(vocab_size=dataset.tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"Training on {DEVICE} with {model.parameter_count():,} parameters")
    best_validation_loss = float("inf")

    for iteration in range(max_iters):
        if iteration % EVAL_INTERVAL == 0 or iteration == max_iters - 1:
            losses = estimate_loss(model, dataset, eval_iters)
            print(
                f"step {iteration:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_validation_loss:
                best_validation_loss = losses["val"]
                save_checkpoint(model, optimizer, iteration, best_validation_loss)

        inputs, targets = dataset.get_batch("train")
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main() -> None:
    """Parse optional short-test settings and start training."""
    parser = argparse.ArgumentParser(description="Train MiniGPT on Tiny Shakespeare.")
    parser.add_argument(
        "--max-iters", type=int, default=MAX_ITERS, help="Number of update steps."
    )
    parser.add_argument(
        "--eval-iters", type=int, default=EVAL_ITERS, help="Batches per evaluation."
    )
    args = parser.parse_args()

    if args.max_iters <= 0 or args.eval_iters <= 0:
        parser.error("--max-iters and --eval-iters must both be positive.")
    train(max_iters=args.max_iters, eval_iters=args.eval_iters)


if __name__ == "__main__":
    main()
