"""
Intrinsic motivation engine for FALL.
Drives autonomous behavior without human prompts.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from collections import defaultdict
import hashlib

class StatePredictor(nn.Module):
    """Predicts next state to measure novelty."""
    def __init__(self, d_model=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
    def forward(self, state_embed):
        return self.net(state_embed)

class IntrinsicMotivation:
    def __init__(self):
        self.state_predictor = StatePredictor()
        self.visit_counts = defaultdict(int)
        self.competence_history = []
        self.novelty_weight = 0.30
        self.competence_weight = 0.25
        self.impact_weight = 0.20
        self.uncertainty_weight = 0.25

    def compute(self, task: str, world_state: Dict) -> float:
        """Compute intrinsic motivation score for a task."""
        r_novelty = self._novelty(task, world_state)
        r_competence = self._competence(task)
        r_impact = self._impact(task, world_state)
        r_uncertainty = self._uncertainty(task)
        total = (
            self.novelty_weight * r_novelty +
            self.competence_weight * r_competence +
            self.impact_weight * r_impact +
            self.uncertainty_weight * r_uncertainty
        )
        return total

    def _novelty(self, task: str, world_state: Dict) -> float:
        """Reward novel states — higher when unexpected."""
        state_hash = self._hash_state(world_state)
        self.visit_counts[state_hash] += 1
        visits = self.visit_counts[state_hash]
        # Higher novelty = fewer visits
        return 1.0 / np.sqrt(visits)

    def _competence(self, task: str) -> float:
        """Reward tasks that match current skill level."""
        if not self.competence_history:
            return 0.5
        recent_success = np.mean(self.competence_history[-100:]) if self.competence_history else 0.5
        # Target ~85% success rate
        return 1.0 - abs(recent_success - 0.85)

    def _impact(self, task: str, world_state: Dict) -> float:
        """Reward tasks that cause meaningful change."""
        # Measure entropy of possible outcomes
        return 0.5  # Placeholder

    def _uncertainty(self, task: str) -> float:
        """Reward exploring where the model is uncertain."""
        return 0.5  # Placeholder — use ensemble disagreement

    def _hash_state(self, state: Dict) -> str:
        state_str = str(sorted(state.items()))
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def record_outcome(self, task: str, success: bool):
        self.competence_history.append(1.0 if success else 0.0)