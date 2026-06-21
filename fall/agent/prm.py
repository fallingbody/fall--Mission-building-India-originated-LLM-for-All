"""
Process Reward Model for FALL.
Scores each reasoning step for correctness.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import re

class ProcessRewardModel(nn.Module):
    def __init__(self, d_model: int = 4096, n_layers: int = 12):
        super().__init__()
        self.d_model = d_model

        # Lightweight transformer for step scoring
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Tokenizer embedding (simplified — uses existing FALL tokenizer)
        self.embed = nn.Embedding(256_000, d_model)

        # Scoring thresholds
        self.correct_threshold = 0.7
        self.unsafe_threshold = 0.3

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Score a sequence of tokens."""
        x = self.embed(token_ids)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.score_head(pooled))

    def score(self, step_text: str) -> float:
        """Score a single reasoning step. Returns 0.0 to 1.0."""
        # Tokenize (placeholder)
        tokens = torch.randint(0, 256000, (1, 512))
        with torch.no_grad():
            score = self.forward(tokens).item()
        return score

    def is_correct(self, step_text: str) -> bool:
        return self.score(step_text) > self.correct_threshold

    def is_unsafe(self, step_text: str) -> bool:
        return self.score(step_text) < self.unsafe_threshold

    def train_on_feedback(self, step_text: str, correct: bool):
        """Update the model based on human or automated feedback."""
        # In production, this would be a proper training loop
        target = 1.0 if correct else 0.0
        tokens = torch.randint(0, 256000, (1, 512))
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        pred = self.forward(tokens)
        loss = F.binary_cross_entropy(pred, torch.tensor([[target]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()