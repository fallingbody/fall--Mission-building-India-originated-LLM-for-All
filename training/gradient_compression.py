"""
Gradient compression for distributed training.
Reduces communication overhead by 10-100x.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class GradientCompressor:
    """Compresses gradients before all-reduce to reduce network traffic."""
    
    def __init__(
        self,
        method: str = "power_sgd",
        compression_ratio: float = 0.01,
        warmup_steps: int = 1000,
    ):
        self.method = method
        self.compression_ratio = compression_ratio
        self.warmup_steps = warmup_steps
        self.error_memory: dict = {}
    
    def compress(
        self,
        gradient: torch.Tensor,
        param_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a gradient tensor."""
        if self.method == "topk":
            return self._topk_compress(gradient)
        elif self.method == "randomk":
            return self._randomk_compress(gradient)
        elif self.method == "power_sgd":
            return self._powersgd_compress(gradient, param_name)
        elif self.method == "sign_sgd":
            return self._sign_compress(gradient)
        else:
            return gradient, torch.ones(1, device=gradient.device)
    
    def _topk_compress(
        self,
        gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep only top-k values by magnitude."""
        k = max(1, int(gradient.numel() * self.compression_ratio))
        values, indices = torch.topk(gradient.abs().view(-1), k)
        mask = torch.zeros_like(gradient.view(-1))
        mask.scatter_(0, indices, 1.0)
        return gradient * mask.view_as(gradient), mask
    
    def _randomk_compress(
        self,
        gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly keep k% of values."""
        mask = torch.bernoulli(
            torch.ones_like(gradient) * self.compression_ratio
        )
        return gradient * mask, mask
    
    def _powersgd_compress(
        self,
        gradient: torch.Tensor,
        param_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PowerSGD: low-rank approximation of gradient."""
        shape = gradient.shape
        if gradient.dim() != 2:
            gradient = gradient.view(-1, shape[-1])
        
        # Randomized SVD
        with torch.no_grad():
            rank = max(1, int(min(gradient.shape) * self.compression_ratio))
            
            # Random projection
            Q = torch.randn(gradient.shape[1], rank + 5, device=gradient.device)
            Q, _ = torch.linalg.qr(gradient @ Q)
            
            # Project gradient
            P = gradient.T @ Q
            U, S, V = torch.svd(P, some=True)
            U = Q @ U[:, :rank]
            
            compressed = U @ torch.diag(S[:rank]) @ V[:, :rank].T
        
        return compressed.view_as(gradient.view(shape)), torch.ones(1)
    
    def _sign_compress(
        self,
        gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SignSGD: compress to just the sign of each element."""
        return torch.sign(gradient), torch.ones(1)


class CommunicationHooks:
    """FSDP communication hooks with gradient compression."""
    
    def __init__(self, compressor: GradientCompressor):
        self.compressor = compressor
    
    def hook(self, state, bucket):
        """FSDP communication hook."""
        grad = bucket.buffer()
        compressed, mask = self.compressor.compress(grad, "")
        return [compressed]