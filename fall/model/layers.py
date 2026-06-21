import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .attention import MultiHeadLatentAttention, SSDBlock, HyperbolicAttention, FourierNeuralOperator
from .moe import AuxiliaryLossFreeMoE

class FALLDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = MultiHeadLatentAttention(config)
        self.moe = AuxiliaryLossFreeMoE(config)
        self.use_mamba = layer_idx in config.use_mamba_layers
        if self.use_mamba:
            self.ssd = SSDBlock(config)
        self.use_hyper = layer_idx in config.use_hyperbolic_layers
        if self.use_hyper:
            self.hyp = HyperbolicAttention(config)
        self.use_fno = layer_idx in config.use_fno_layers
        if self.use_fno:
            self.fno = FourierNeuralOperator(config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model) if (self.use_hyper or self.use_fno) else None

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        if self.use_hyper:
            x = x + self.hyp(self.norm3(x), mask)
        if self.use_fno:
            x = x + self.fno(self.norm3(x))
        if self.use_mamba:
            x = x + self.ssd(self.norm2(x))
        else:
            # MoE with gradient checkpointing
            if self.training and getattr(self, '_checkpoint', True) and torch.is_grad_enabled():
                x = x + checkpoint(self.moe, self.norm2(x))
            else:
                x = x + self.moe(self.norm2(x))
        return x