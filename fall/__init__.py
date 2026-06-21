"""
FALL - Fully Autonomous Learning Language model
5 Trillion Parameters | For All
"""
__version__ = "1.0.0"
__author__ = "FALL Development Team"

from fall.model.config import FALLConfig
from fall.model.model import FALLForCausalLM
from fall.training.trainer import FALLTrainer
from fall.inference.api import FALLInferenceServer
from fall.agent.runtime import FALLAutonomousAgent

__all__ = [
    "FALLConfig",
    "FALLForCausalLM",
    "FALLTrainer",
    "FALLInferenceServer",
    "FALLAutonomousAgent",
]