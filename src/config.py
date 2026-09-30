"""Central configuration for the educational MiniGPT project.

Keep tunable values here instead of scattering magic numbers through the
model, dataset, training, and generation scripts.
"""

from pathlib import Path

import torch


# Project paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "input.txt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "minigpt.pt"


# Runtime ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# Tokenizer --------------------------------------------------------------------
# GPT-2 BPE always has this fixed vocabulary. Tokenizer IDs range from 0 to
# VOCAB_SIZE - 1, and the model predicts one score for every possible ID.
VOCAB_SIZE = 50_257


# Dataset and training ---------------------------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 128  # Maximum number of preceding tokens the model can use.
MAX_ITERS = 5_000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1


# Transformer architecture -----------------------------------------------------
N_EMBD = 128      # Size of each token/position embedding vector.
N_HEAD = 4        # Parallel attention heads per transformer block.
N_LAYER = 4       # Number of stacked transformer blocks.
DROPOUT = 0.2

if N_EMBD % N_HEAD != 0:
    raise ValueError("N_EMBD must be divisible by N_HEAD.")


# Text generation --------------------------------------------------------------
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 40


def set_seed(seed: int = SEED) -> None:
    """Seed PyTorch's random generators for repeatable experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Print the active configuration for a quick setup check."""
    print("MiniGPT configuration")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Context length: {BLOCK_SIZE}")
    print(f"Model: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} embeddings")
    print(f"Vocabulary size: {VOCAB_SIZE:,}")


if __name__ == "__main__":
    main()
