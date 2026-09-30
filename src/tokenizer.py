"""Tokenization helpers for MiniGPT.

The model operates on integer token IDs, not raw text. This module wraps the
GPT-2 byte-pair encoding (BPE) tokenizer supplied by ``tiktoken``.
"""

from collections.abc import Sequence

import tiktoken


class GPTTokenizer:
    """Convert between text and the GPT-2 BPE token-ID vocabulary."""

    def __init__(self) -> None:
        """Load the fixed GPT-2 BPE vocabulary once for this tokenizer."""
        self._encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        """Return the integer token IDs representing ``text``.

        One token is not necessarily one word or one character. BPE commonly
        represents frequent word pieces with a single integer ID.
        """
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, token_ids: Sequence[int]) -> str:
        """Reconstruct text from a sequence of integer token IDs."""
        return self._encoding.decode(list(token_ids))

    @property
    def vocab_size(self) -> int:
        """Return the number of IDs the model can predict: 50,257 for GPT-2."""
        return self._encoding.n_vocab


def main() -> None:
    """Run a small round-trip test when this module is executed directly."""
    tokenizer = GPTTokenizer()
    sample = "Hello! Welcome to MiniGPT."
    token_ids = tokenizer.encode(sample)

    print(f"Original text: {sample}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {tokenizer.decode(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    assert tokenizer.decode(token_ids) == sample, "Tokenization round-trip failed."
    print("Round-trip test passed.")


if __name__ == "__main__":
    main()
