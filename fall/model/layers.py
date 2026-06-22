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
            self.fno = FourierNeuralOperator(config.d_model, n_modes=getattr(config, 'fno_n_modes', 16))
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model) if (self.use_hyper or self.use_fno) else None

    def forward(self, x, mask=None, is_reasoning_mode=False, is_gradient_checkpointing=False):
        if is_gradient_checkpointing:
            x = x + torch.utils.checkpoint.checkpoint(self.attn, self.norm1(x), mask, use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x), mask)
            
        if self.use_hyper:
            if is_gradient_checkpointing:
                x = x + torch.utils.checkpoint.checkpoint(self.hyp, self.norm3(x), mask, use_reentrant=False)
            else:
                x = x + self.hyp(self.norm3(x), mask)
        if self.use_fno:
            if is_gradient_checkpointing:
                x = x + torch.utils.checkpoint.checkpoint(self.fno, self.norm3(x), use_reentrant=False)
            else:
                x = x + self.fno(self.norm3(x))
        if self.use_mamba:
            if is_gradient_checkpointing:
                x = x + torch.utils.checkpoint.checkpoint(self.ssd, self.norm2(x), use_reentrant=False)
            else:
                x = x + self.ssd(self.norm2(x))
        else:
            # MoE without gradient checkpointing (prevents shape mismatch during backward)
            x = x + self.moe(self.norm2(x), is_reasoning_mode=is_reasoning_mode)
        return x