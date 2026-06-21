"""Pytest configuration for FALL."""
import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    