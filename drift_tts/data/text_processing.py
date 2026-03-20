"""UTF-8 byte-level text tokenization.

Inspired by Echo TTS: text is encoded as raw UTF-8 bytes (vocab size = 256).
This allows natural handling of any language and special characters without
a trained tokenizer model.

Special tokens:
  0 = PAD
  1 = BOS
  2 = EOS
  3-255 = UTF-8 byte values (shifted by +3)
"""

from typing import List, Tuple

import torch

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 259  # 256 byte values + PAD + BOS + EOS


class UTF8Tokenizer:
    """UTF-8 byte-level tokenizer (no trained model needed)."""

    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.pad_token_id = PAD_TOKEN
        self.bos_token_id = BOS_TOKEN
        self.eos_token_id = EOS_TOKEN

    def encode(self, text: str) -> List[int]:
        """Encode text to UTF-8 byte token IDs."""
        byte_ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]
        return [self.bos_token_id] + byte_ids + [self.eos_token_id]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_vals = []
        for i in ids:
            if i in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            byte_vals.append(i - BYTE_OFFSET)
        return bytes(byte_vals).decode("utf-8", errors="replace")

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts with padding.

        Returns:
            token_ids: [B, L] padded token IDs.
            mask: [B, L] attention mask (1 = valid, 0 = pad).
        """
        batch_ids = [self.encode(t)[:max_length] for t in texts]
        lengths = [len(ids) for ids in batch_ids]
        max_len = min(max(lengths), max_length)

        token_ids = torch.full((len(batch_ids), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, : len(ids)] = 1
        return token_ids, mask
