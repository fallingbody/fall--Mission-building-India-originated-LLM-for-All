"""
Curriculum learning for FALL.
Progressively increases difficulty during training.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CurriculumPhase:
    name: str
    start_step: int
    end_step: int
    seq_len: int
    data_mix: Dict[str, float]
    difficulty: float

class CurriculumScheduler:
    def __init__(self):
        self.phases = [
            CurriculumPhase(
                name="warmup",
                start_step=0,
                end_step=5_000,
                seq_len=8192,
                data_mix={"code": 0.5, "web": 0.3, "docs": 0.2},
                difficulty=0.1,
            ),
            CurriculumPhase(
                name="short_context",
                start_step=5_000,
                end_step=100_000,
                seq_len=65536,
                data_mix={"code": 0.6, "web": 0.2, "security": 0.1, "docs": 0.1},
                difficulty=0.3,
            ),
            CurriculumPhase(
                name="medium_context",
                start_step=100_000,
                end_step=500_000,
                seq_len=262144,
                data_mix={"code": 0.4, "web": 0.2, "security": 0.2, "docs": 0.1, "synthetic": 0.1},
                difficulty=0.5,
            ),
            CurriculumPhase(
                name="long_context",
                start_step=500_000,
                end_step=1_500_000,
                seq_len=524288,
                data_mix={"code": 0.3, "web": 0.15, "security": 0.25, "docs": 0.1, "synthetic": 0.2},
                difficulty=0.7,
            ),
            CurriculumPhase(
                name="full_context",
                start_step=1_500_000,
                end_step=3_750_000,
                seq_len=524288,
                data_mix={"code": 0.25, "web": 0.1, "security": 0.3, "docs": 0.05, "synthetic": 0.3},
                difficulty=1.0,
            ),
        ]
    
    def get_phase(self, step: int) -> CurriculumPhase:
        for phase in self.phases:
            if phase.start_step <= step < phase.end_step:
                return phase
        return self.phases[-1]
    
    def get_seq_len(self, step: int) -> int:
        return self.get_phase(step).seq_len
    
    def get_data_mix(self, step: int) -> Dict[str, float]:
        return self.get_phase(step).data_mix