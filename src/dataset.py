"""Load Tiny Shakespeare and create next-token prediction batches."""

from pathlib import Path
from typing import Literal

import torch

try:  # Supports both `python src/dataset.py` and `python -m src.dataset`.
    from .config import BATCH_SIZE, BLOCK_SIZE, DEVICE
    from .tokenizer import GPTTokenizer
except ImportError:
    from config import BATCH_SIZE, BLOCK_SIZE, DEVICE
    from tokenizer import GPTTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "input.txt"


class TextDataset:
    """An in-memory token stream with randomly sampled language-model batches."""

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH) -> None:
        """Read, tokenize, and split the source text into train and validation data.

        The split preserves the original text order: the first 90% is training
        data and the final 10% is validation data. A token is stored as
        ``torch.long`` because PyTorch embeddings require integer index tensors.
        """
        self.tokenizer = GPTTokenizer()
        self.data_path = Path(data_path)

        if not self.data_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        text = self.data_path.read_text(encoding="utf-8")
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        split_index = int(0.9 * len(self.data))
        self.train_data = self.data[:split_index]
        self.val_data = self.data[split_index:]

        print(f"Loaded {len(self.data):,} tokens from {self.data_path.name}")
        print(f"Training tokens:   {len(self.train_data):,}")
        print(f"Validation tokens: {len(self.val_data):,}")

    def get_batch(
        self, split: Literal["train", "val"] = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one random batch of input tokens and one-token-shifted targets.

        ``x[b]`` might contain tokens ``[A, B, C]`` and ``y[b]`` then contains
        ``[B, C, D]``. During training, MiniGPT learns to predict every target
        token from the preceding context at the same position in ``x``.
        """
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        data = self.train_data if split == "train" else self.val_data
        if len(data) <= BLOCK_SIZE:
            raise ValueError(
                f"The {split} split needs more than BLOCK_SIZE={BLOCK_SIZE} tokens."
            )

        start_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in start_indices])
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in start_indices])

        return x.to(DEVICE), y.to(DEVICE)


def main() -> None:
    """Run a shape and next-token alignment test."""
    dataset = TextDataset()
    inputs, targets = dataset.get_batch()

    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")

    # The target at position t must be the input at position t + 1.
    assert torch.equal(inputs[0, 1:].cpu(), targets[0, :-1].cpu())
    print("Batch alignment test passed.")


if __name__ == "__main__":
    main()
