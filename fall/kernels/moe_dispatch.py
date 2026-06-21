"""
MoE Token Dispatch and Combine - Python reference implementation.
Used for testing and CPU fallback when CUDA kernels are unavailable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def moe_dispatch(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    num_experts: int,
    top_k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to experts based on router decisions.
    
    Args:
        hidden_states: (batch * seq_len, d_model) flattened input
        router_indices: (batch * seq_len, top_k) expert indices per token
        router_weights: (batch * seq_len, top_k) routing weights per token
        num_experts: total number of experts
        top_k: number of experts per token
    
    Returns:
        expert_inputs: (num_experts, max_tokens, d_model) inputs per expert
        expert_counts: (num_experts,) number of tokens per expert
    """
    B, D = hidden_states.shape
    device = hidden_states.device
    
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    for k in range(top_k):
        expert_ids = router_indices[:, k]
        expert_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.long))
    
    max_tokens = expert_counts.max().item()
    
    # Allocate output buffer
    expert_inputs = torch.zeros(num_experts, max_tokens, D, dtype=hidden_states.dtype, device=device)
    
    # Track position per expert
    expert_positions = torch.zeros(num_experts, dtype=torch.long, device=device)
    
    # Scatter tokens
    for token_idx in range(B):
        for k in range(top_k):
            expert_id = router_indices[token_idx, k].item()
            weight = router_weights[token_idx, k]
            pos = expert_positions[expert_id].item()
            
            expert_inputs[expert_id, pos] = hidden_states[token_idx] * weight
            expert_positions[expert_id] += 1
    
    return expert_inputs, expert_counts


def moe_combine(
    expert_outputs: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    expert_token_counts: torch.Tensor,
    d_model: int,
    top_k: int = 2,
) -> torch.Tensor:
    """
    Combine expert outputs back into the original token order.
    
    Args:
        expert_outputs: (num_experts, max_tokens, d_model) outputs per expert
        router_indices: (batch * seq_len, top_k) expert indices per token
        router_weights: (batch * seq_len, top_k) routing weights per token
        expert_token_counts: (num_experts,) number of tokens per expert
        d_model: model dimension
        top_k: number of experts per token
    
    Returns:
        combined: (batch * seq_len, d_model) combined output
    """
    num_tokens = router_indices.shape[0]
    device = expert_outputs.device
    
    combined = torch.zeros(num_tokens, d_model, dtype=expert_outputs.dtype, device=device)
    
    # Track position per expert
    expert_positions = torch.zeros(len(expert_token_counts), dtype=torch.long, device=device)
    
    # Gather outputs
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = router_indices[token_idx, k].item()
            weight = router_weights[token_idx, k]
            pos = expert_positions[expert_id].item()
            
            combined[token_idx] += expert_outputs[expert_id, pos] * weight
            expert_positions[expert_id] += 1
    
    return combined


class MoEDispatcher(nn.Module):
    """
    Pure PyTorch MoE dispatch module.
    Handles token routing, expert computation, and output combination.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router_noise = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with token routing.
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: load balancing loss
        """
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        
        # Compute routing probabilities
        clean_logits = self.router(x_flat)
        
        if self.training:
            noise_std = F.softplus(self.router_noise(x_flat)) + 1e-2
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            noisy_logits = clean_logits
        
        routing_probs = F.softmax(noisy_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Dispatch
        expert_inputs, expert_counts = moe_dispatch(
            x_flat, topk_indices, topk_probs, self.num_experts, self.top_k
        )
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(expert_inputs)
        for e_idx, expert in enumerate(self.experts):
            if expert_counts[e_idx] > 0:
                expert_outputs[e_idx, :expert_counts[e_idx]] = expert(
                    expert_inputs[e_idx, :expert_counts[e_idx]]
                )
        
        # Combine
        combined = moe_combine(
            expert_outputs, topk_indices, topk_probs,
            expert_counts, D, self.top_k
        )
        
        # Load balancing loss
        f_i = expert_counts.float() / (B * L * self.top_k + 1e-8)
        P_i = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        return combined.view(B, L, D), aux_loss


def moe_dispatch_fast(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized dispatch using scatter operations.
    Faster than the loop-based version for large batches.
    """
    B, D = hidden_states.shape
    top_k = router_indices.shape[1]
    device = hidden_states.device
    
    # Weight tokens by routing probabilities
    weighted = hidden_states.unsqueeze(1) * router_weights.unsqueeze(-1)  # (B, top_k, D)
    
    # Count per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    for k in range(top_k):
        expert_counts.scatter_add_(0, router_indices[:, k], torch.ones(B, dtype=torch.long, device=device))
    
    max_tokens = expert_counts.max().item()
    
    # Allocate
    expert_inputs = torch.zeros(num_experts, max_tokens, D, dtype=hidden_states.dtype, device=device)
    positions = torch.zeros(num_experts, dtype=torch.long, device=device)
    
    # Scatter using index_put for efficiency
    for k in range(top_k):
        for token_idx in range(B):
            eid = router_indices[token_idx, k].item()
            pos = positions[eid].item()
            expert_inputs[eid, pos] = weighted[token_idx, k]
            positions[eid] += 1
    
    return expert_inputs, expert_counts