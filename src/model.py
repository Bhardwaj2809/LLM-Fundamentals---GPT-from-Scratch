"""The MiniGPT decoder-only Transformer model.

The model receives token IDs and predicts a probability distribution over the
next token at every position in the input sequence.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

try:  # Supports both `python src/model.py` and `python -m src.model`.
    from .config import BLOCK_SIZE, DROPOUT, N_EMBD, N_HEAD, N_LAYER, VOCAB_SIZE
except ImportError:
    from config import BLOCK_SIZE, DROPOUT, N_EMBD, N_HEAD, N_LAYER, VOCAB_SIZE


class Head(nn.Module):
    """One masked self-attention head.

    Each position creates a query, key, and value vector. Query-key similarity
    determines attention weights; the causal mask prevents a token from seeing
    tokens that occur later in the sequence.
    """

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        # A buffer is saved with the model but is not a trainable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix information from the current and earlier positions in ``x``."""
        _, time_steps, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)

        # Scale prevents dot products from growing too large as head size grows.
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * (keys.size(-1) ** -0.5)
        attention_scores = attention_scores.masked_fill(
            self.tril[:time_steps, :time_steps] == 0, float("-inf")
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        values = self.value(x)
        return attention_weights @ values


class MultiHeadAttention(nn.Module):
    """Run several attention heads in parallel, then recombine their outputs."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a full embedding-sized attention update for each token."""
        combined_heads = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined_heads))


class FeedForward(nn.Module):
    """The position-wise MLP that transforms each token independently.

    Attention shares information across positions. This network then performs
    richer nonlinear computation on the information gathered at each position.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform each embedding while preserving its shape."""
        return self.net(x)


class Block(nn.Module):
    """One pre-norm Transformer block: attention and MLP residual updates."""

    def __init__(self) -> None:
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.layer_norm_1 = nn.LayerNorm(N_EMBD)
        self.attention = MultiHeadAttention(N_HEAD, head_size)
        self.layer_norm_2 = nn.LayerNorm(N_EMBD)
        self.feed_forward = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalized sublayers and preserve information through skips."""
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """A small decoder-only GPT language model for next-token prediction."""

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*(Block() for _ in range(N_LAYER)))
        self.final_layer_norm = nn.LayerNorm(N_EMBD)
        self.language_model_head = nn.Linear(N_EMBD, vocab_size)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and, when targets are supplied, next-token loss.

        Args:
            token_ids: Integer tensor of shape ``(batch, time)``.
            targets: Shifted token IDs of the same shape, used only for training.
        """
        _, time_steps = token_ids.shape
        if time_steps > BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {time_steps} exceeds BLOCK_SIZE={BLOCK_SIZE}."
            )

        positions = torch.arange(time_steps, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample and append up to ``max_new_tokens`` IDs.

        Temperature controls randomness: below 1.0 makes choices sharper; above
        1.0 makes them more varied. Top-k keeps only the k highest-scoring IDs.
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        for _ in range(max_new_tokens):
            context = token_ids[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be greater than zero or None")
                top_values, _ = torch.topk(
                    next_token_logits, min(top_k, self.vocab_size)
                )
                cutoff = top_values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff, float("-inf")
                )

            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids

    def parameter_count(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def main() -> None:
    """Run a small forward-pass and loss-shape test."""
    model = GPTLanguageModel()
    example_tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits, loss = model(example_tokens, example_tokens)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert loss is not None and math.isfinite(loss.item())
    print(f"Logit shape: {tuple(logits.shape)}")
    print(f"Test loss: {loss.item():.4f}")
    print(f"Trainable parameters: {model.parameter_count():,}")
    print("Model forward-pass test passed.")


if __name__ == "__main__":
    main()
