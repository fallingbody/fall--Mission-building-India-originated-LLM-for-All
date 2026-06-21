"""
Python bindings for FALL custom CUDA kernels.
Uses torch.utils.cpp_extension for JIT compilation.
"""
import torch
from torch.utils.cpp_extension import load
import os

_kernel_dir = os.path.dirname(os.path.abspath(__file__))

# Compile CUDA kernels on import
_attention_kernels = load(
    name="fall_attention",
    sources=[
        os.path.join(_kernel_dir, "attention_fwd.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_90,code=sm_90",  # H100
        "-gencode=arch=compute_89,code=sm_89",  # Ada
        "-gencode=arch=compute_80,code=sm_80",  # A100
    ],
    verbose=False,
)

_moe_kernels = load(
    name="fall_moe",
    sources=[
        os.path.join(_kernel_dir, "moe_dispatch.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

_ssd_kernels = load(
    name="fall_ssd",
    sources=[
        os.path.join(_kernel_dir, "ssd_scan.cu"),
    ],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

_rope_kernels = load(
    name="fall_rope",
    sources=[
        os.path.join(_kernel_dir, "rope_fused.cu"),
    ],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class FusedMLAAttention(torch.autograd.Function):
    """Fused MLA attention forward."""
    
    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, L, D = q.shape
        O = torch.empty_like(q)
        Lse = torch.empty(B, H, L, device=q.device)
        
        _attention_kernels.mla_attention_fwd(
            q, k, v, O, Lse,
            B, H, L, D, scale,
        )
        
        ctx.save_for_backward(q, k, v, O, Lse)
        ctx.scale = scale
        return O
    
    @staticmethod
    def backward(ctx, dO):
        # Backward pass (omitted for brevity)
        pass


class FusedMoEDispatch(torch.autograd.Function):
    """Fused MoE token dispatch and combine."""
    
    @staticmethod
    def forward(ctx, x, router_idx, router_weights, experts):
        B, L, D = x.shape
        N = B * L
        E = len(experts)
        
        expert_inputs = torch.zeros(E, N, D, device=x.device, dtype=x.dtype)
        expert_counts = torch.zeros(E, device=x.device, dtype=torch.int32)
        
        _moe_kernels.moe_dispatch(
            x.view(-1, D),
            router_idx.view(-1, router_idx.shape[-1]),
            router_weights.view(-1, router_weights.shape[-1]),
            expert_inputs.view(E, -1),
            expert_counts,
            N, D, E, router_idx.shape[-1],
        )
        
        return expert_inputs, expert_counts


def fused_mla_attention(q, k, v, scale=None):
    """Fused MLA attention forward pass."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return FusedMLAAttention.apply(q, k, v, scale)


def fused_moe(x, router_idx, router_weights, experts):
    """Fused MoE dispatch."""
    return FusedMoEDispatch.apply(x, router_idx, router_weights, experts)