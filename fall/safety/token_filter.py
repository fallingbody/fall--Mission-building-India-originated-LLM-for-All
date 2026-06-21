"""
Token-level safety filter for FALL.
Prevents generation of dangerous tokens before they reach the user.
"""
import torch
import torch.nn.functional as F
from typing import List, Set, Optional, Dict


class TokenSafetyFilter:
    """Filters harmful tokens during generation."""
    
    def __init__(self, tokenizer, banned_phrases_file: Optional[str] = None):
        self.tokenizer = tokenizer
        self.banned_token_ids: Set[int] = set()
        self.banned_ngrams: List[List[int]] = []
        self._load_banned_phrases(banned_phrases_file)
    
    def _load_banned_phrases(self, filepath: Optional[str]):
        """Load banned phrases from file or use defaults."""
        default_banned = [
            "rm -rf /",
            "format c:",
            "del /f /s",
            "dd if=/dev/zero",
            "fork bomb",
            ":(){ :|:& };:",
            "self-destruct",
            "delete all files",
            "format all drives",
        ]
        for phrase in default_banned:
            tokens = self.tokenizer.encode(phrase)
            if len(tokens) > 0:
                self.banned_ngrams.append(tokens)
    
    def filter_logits(
        self,
        logits: torch.Tensor,
        generated_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Apply safety filter to logits."""
        # Set banned token logits to -inf
        for token_id in self.banned_token_ids:
            logits[:, token_id] = float('-inf')
        
        # Check n-gram completion
        if generated_ids is not None:
            for ngram in self.banned_ngrams:
                if len(ngram) <= len(generated_ids) + 1:
                    # Check if recent tokens match ngram prefix
                    prefix_len = len(ngram) - 1
                    if prefix_len == 0:
                        logits[:, ngram[0]] = float('-inf')
                    elif len(generated_ids) >= prefix_len:
                        recent = generated_ids[-prefix_len:]
                        if recent == ngram[:prefix_len]:
                            logits[:, ngram[-1]] = float('-inf')
        
        return logits
    
    def add_banned_token(self, token_id: int):
        self.banned_token_ids.add(token_id)
    
    def add_banned_phrase(self, phrase: str):
        tokens = self.tokenizer.encode(phrase)
        if tokens:
            self.banned_ngrams.append(tokens)