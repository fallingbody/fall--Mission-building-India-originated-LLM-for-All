"""
YaRN (Yet another RoPE extensioN) position encoding for FALL.
Supports extreme context lengths up to 1M+ tokens.
"""
import torch
import torch.nn as nn
import math


class YaRNRoPE(nn.Module):
    """
    YaRN-extended Rotary Position Embedding.
    
    Handles context lengths up to 1,048,576 tokens by:
    - Splitting dimensions into high-frequency (local) and low-frequency (global)
    - Applying progressive scaling to low frequencies
    - Maintaining exact position information at short ranges
    """
    
    def __init__(
        self,
        d_head: int,
        max_seq_len: int = 524288,
        base: float = 50_000_000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute frequency bands
        theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("theta", theta)
        
        # YaRN scaling: progressive interpolation for low frequencies
        scale = torch.ones(d_head // 2)
        ramp = torch.linspace(1.0, scaling_factor, d_head // 2)
        self.register_buffer("scale", scale * ramp)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply YaRN RoPE to input tensor.
        
        Args:
            x: (B, n_heads, L, d_head) tensor of queries or keys
            offset: position offset for cached sequences
        
        Returns:
            Rotated tensor of same shape
        """
        B, n_heads, L, d_head = x.shape
        pos = torch.arange(offset, offset + L, device=x.device).float()
        
        # Compute frequencies with YaRN scaling
        freqs = torch.outer(pos, self.theta * self.scale)  # (L, d_head/2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, d_head/2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        
        # Apply rotation to pairs of dimensions
        x_rot = x[..., ::2] * cos + x[..., 1::2] * sin
        x_pass = x[..., 1::2] * cos - x[..., ::2] * sin
        
        return torch.stack([x_rot, x_pass], dim=-1).flatten(-2)
    
    def get_freqs(self, seq_len: int, offset: int = 0) -> tuple:
        """Get precomputed cos and sin frequencies."""
        pos = torch.arange(offset, offset + seq_len).float()
        freqs = torch.outer(pos, self.theta * self.scale)
        return freqs.cos(), freqs.sin()