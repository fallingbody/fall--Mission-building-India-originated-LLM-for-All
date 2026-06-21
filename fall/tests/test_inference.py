"""Integration tests for FALL inference."""
import pytest
import asyncio
from fall.inference.server import FALLInferenceServer, InferenceConfig

class TestInference:
    @pytest.fixture
    def config(self):
        return InferenceConfig(model_path="/models/fall_test")
    
    def test_server_creation(self, config):
        server = FALLInferenceServer(config)
        assert server is not None
        assert server.stats["total_requests"] == 0