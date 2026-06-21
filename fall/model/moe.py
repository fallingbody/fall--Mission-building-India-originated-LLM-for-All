import torch
import torch.nn as nn
import torch.nn.functional as F

class KANExpert(nn.Module):
    # Kolmogorov-Arnold Network expert (simplified)
    def __init__(self, d_model, d_hidden, grid_size=8):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.coeffs1 = nn.Parameter(torch.randn(d_model, d_hidden, grid_size+3) * 0.1)
        self.coeffs2 = nn.Parameter(torch.randn(d_hidden, d_model, grid_size+3) * 0.1)
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # Simplified KAN forward (actual implementation would use B‑spline basis)
        return F.gelu(x @ self.coeffs1.mean(dim=-1)) @ self.coeffs2.mean(dim=-1)

class MLPExpert(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_ffn, d_model, bias=False)
        )
    def forward(self, x):
        return self.net(x)

class AuxiliaryLossFreeMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts_per_layer
        self.top_k = config.n_active_experts
        self.router = nn.Linear(config.d_model, self.n_experts, bias=True)
        self.expert_bias = nn.Parameter(torch.zeros(self.n_experts))
        # Create experts, mixing MLP and KAN
        self.experts = nn.ModuleList()
        kan_count = int(self.n_experts * config.kan_expert_ratio)
        for i in range(self.n_experts):
            if i < kan_count:
                self.experts.append(KANExpert(config.d_model, config.d_model // 2))
            else:
                self.experts.append(MLPExpert(config.d_model, config.d_ffn))

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat) + self.expert_bias
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.experts):
            mask = (topk_idx == e_idx).any(dim=-1)
            if mask.any():
                e_prob = topk_probs[mask][topk_idx[mask] == e_idx].unsqueeze(-1)
                out[mask] += expert(x_flat[mask]) * e_prob
        # Dynamic bias adjustment (no auxiliary loss)
        with torch.no_grad():
            tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
            tokens_per_expert.scatter_add_(0, topk_idx.flatten(), torch.ones_like(topk_idx.flatten(), dtype=torch.float))
            avg = tokens_per_expert.mean()
            overloaded = tokens_per_expert > 1.5 * avg
            underloaded = tokens_per_expert < 0.5 * avg
            self.expert_bias.data[overloaded] -= 0.01
            self.expert_bias.data[underloaded] += 0.01
        return out.view(B, L, D)