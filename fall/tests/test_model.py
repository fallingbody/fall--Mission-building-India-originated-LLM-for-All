"""
Unit tests for FALL model architecture.
"""
import torch
import pytest
from fall.model.config import FALLConfig
from fall.model.model import FALLForCausalLM
from fall.model.attention import (
    MultiHeadLatentAttention,
    SSDBlock,
    HyperbolicAttention,
    FourierNeuralOperator,
)
from fall.model.moe import AuxiliaryLossFreeMoE, KANExpert, MLPExpert

class TestFALLModel:
    @pytest.fixture
    def config(self):
        return FALLConfig()
    
    @pytest.fixture
    def small_config(self):
        config = FALLConfig()
        config.d_model = 512
        config.n_layers = 4
        config.n_heads = 8
        config.d_head = 64
        config.d_ffn = 1024
        config.n_experts_per_layer = 8
        config.max_seq_len = 2048
        return config
    
    def test_model_creation(self, small_config):
        model = FALLForCausalLM(small_config)
        assert model is not None
        assert len(model.layers) == small_config.n_layers
    
    def test_forward_pass(self, small_config):
        model = FALLForCausalLM(small_config)
        input_ids = torch.randint(0, 1000, (2, 128))
        logits = model(input_ids)
        assert logits.shape == (2, 128, small_config.vocab_size)
    
    def test_attention_shapes(self, small_config):
        attn = MultiHeadLatentAttention(small_config)
        x = torch.randn(2, 64, small_config.d_model)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_ssd_block(self, small_config):
        ssd = SSDBlock(small_config)
        x = torch.randn(2, 64, small_config.d_model)
        out = ssd(x)
        assert out.shape == x.shape
    
    def test_hyperbolic_attention(self, small_config):
        hyp = HyperbolicAttention(small_config)
        x = torch.randn(2, 64, small_config.d_model)
        out = hyp(x)
        assert out.shape == x.shape
    
    def test_fno(self, small_config):
        fno = FourierNeuralOperator(small_config.d_model)
        x = torch.randn(2, 128, small_config.d_model)
        out = fno(x)
        assert out.shape == x.shape
    
    def test_moe(self, small_config):
        moe = AuxiliaryLossFreeMoE(small_config)
        x = torch.randn(2, 32, small_config.d_model)
        out = moe(x)
        assert out.shape == x.shape
    
    def test_kan_expert(self, small_config):
        kan = KANExpert(small_config.d_model, small_config.d_model // 2)
        x = torch.randn(4, 16, small_config.d_model)
        out = kan(x)
        assert out.shape == x.shape
    
    def test_gradient_flow(self, small_config):
        model = FALLForCausalLM(small_config)
        input_ids = torch.randint(0, 1000, (2, 64))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"