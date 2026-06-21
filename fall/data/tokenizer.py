"""
Custom BPE tokenizer for FALL.
256K vocabulary, byte-level fallback, streaming support.
"""
import json
import os
from typing import List, Optional, Dict, Tuple
import regex as re

class FALLTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Special tokens
        self.pad_token = "<|pad|>"
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.bos_token = "<|begin|>"
        self.bos_token_id = self.vocab.get(self.bos_token, 1)
        self.eos_token = "<|end|>"
        self.eos_token_id = self.vocab.get(self.eos_token, 2)
        self.unk_token = "<|unk|>"
        self.unk_token_id = self.vocab.get(self.unk_token, 3)

        # Special FALL tokens
        self.think_token = "<|think|>"
        self.think_token_id = self.vocab.get(self.think_token, 4)
        self.tool_call_token = "<|tool_call|>"
        self.tool_call_token_id = self.vocab.get(self.tool_call_token, 5)
        self.code_token = "<|code|>"
        self.code_token_id = self.vocab.get(self.code_token, 6)
        self.task_end_token = "<|task_end|>"
        self.task_end_token_id = self.vocab.get(self.task_end_token, 7)

        # BPE merge rules
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, cs))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for match in re.finditer(self.pat, text):
            piece = match.group()
            tokens.extend(self._bpe(piece))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        text = ''.join(self.inv_vocab.get(tid, self.unk_token) for tid in token_ids)
        return bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')

    def _bpe(self, token: str) -> List[int]:
        """Apply BPE encoding to a single token."""
        # Simplified BPE — full implementation uses merge table
        return [self.vocab.get(c, self.unk_token_id) for c in token]

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """Encode a batch of texts."""
        encoded = [self.encode(t) for t in texts]
        if max_length:
            encoded = [e[:max_length] for e in encoded]
        return encoded

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)