import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba
from fall.model.position import YaRNRoPE

# ---------- MLA ----------
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_latent = config.d_latent_kv
        self.d_qk_nope = 64
        self.d_qk_rope = 64

        self.kv_compress = nn.Linear(config.d_model, self.d_latent, bias=False)
        self.k_proj = nn.Linear(self.d_latent, config.n_heads * self.d_qk_nope, bias=False)
        self.v_proj = nn.Linear(self.d_latent, config.n_heads * config.d_head, bias=False)
        self.q_proj_nope = nn.Linear(config.d_model, config.n_heads * self.d_qk_nope, bias=False)
        self.q_proj_rope = nn.Linear(config.d_model, config.n_heads * self.d_qk_rope, bias=False)
        self.k_proj_rope = nn.Linear(config.d_model, config.n_heads * self.d_qk_rope, bias=False)
        self.rope = YaRNRoPE(self.d_qk_rope, config.max_seq_len, config.rope_base)
        self.out_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        if config.use_differential_attn:
            self.lambda_init = 0.8
            self.lambda_q1 = nn.Linear(config.d_model, config.n_heads, bias=False)
            self.lambda_k1 = nn.Linear(config.d_model, config.n_heads, bias=False)
            self.lambda_q2 = nn.Linear(config.d_model, config.n_heads, bias=False)
            self.lambda_k2 = nn.Linear(config.d_model, config.n_heads, bias=False)
        self.use_differential = config.use_differential_attn

    def forward(self, x, mask=None):
        B, L, D = x.shape
        n = self.n_heads

        # KV latent
        c_kv = self.kv_compress(x)
        k_nope = self.k_proj(c_kv).view(B, L, n, self.d_qk_nope)
        v = self.v_proj(c_kv).view(B, L, n, self.d_head)

        # Q
        q_nope = self.q_proj_nope(x).view(B, L, n, self.d_qk_nope)
        q_rope = self.q_proj_rope(x).view(B, L, n, self.d_qk_rope)
        k_rope = self.k_proj_rope(x).view(B, L, n, self.d_qk_rope)

        # RoPE on positional parts
        q_rope = self.rope(q_rope.transpose(1,2)).transpose(1,2)
        k_rope = self.rope(k_rope.transpose(1,2)).transpose(1,2)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Attention
        if self.use_differential:
            return self._differential_attn(q, k, v, mask)
        else:
            scale = (self.d_qk_nope + self.d_qk_rope) ** -0.5
            attn = torch.einsum('b l n d, b m n d -> b n l m', q, k) * scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('b n l m, b m n d -> b l n d', attn, v)
            out = out.reshape(B, L, n * self.d_head)
            return self.out_proj(out)

    def _differential_attn(self, q, k, v, mask):
        B, L, n, d = q.shape
        scale = d ** -0.5
        # Attention map 1
        attn1 = torch.einsum('b l n d, b m n d -> b n l m', q, k) * scale
        # Attention map 2 with learned gating
        lambda1 = torch.sigmoid(self.lambda_q1(q) + self.lambda_k1(k)).mean(dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        lambda2 = torch.sigmoid(self.lambda_q2(q) + self.lambda_k2(k)).mean(dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        attn2 = torch.einsum('b l n d, b m n d -> b n l m', q * lambda1, k * lambda2) * scale
        attn = attn1 - self.lambda_init * attn2
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('b n l m, b m n d -> b l n d', attn, v)
        out = out.reshape(B, L, n * self.d_head)
        return self.out_proj(out)

# ---------- SSD (Mamba-2) ----------
class SSDBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mamba = Mamba(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv)
    def forward(self, x):
        return self.mamba(x)

# ---------- Hyperbolic Attention ----------
class HyperbolicAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = 4
        self.d_head = config.d_model // self.n_heads
        self.curvature = nn.Parameter(torch.tensor(-1.0))
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        # Exponential map to Poincaré ball
        q_hyp = self._exp_map(q)
        k_hyp = self._exp_map(k)
        dist = self._hyp_dist(q_hyp, k_hyp)  # (B, n, L, L)
        attn = F.softmax(-dist / math.sqrt(self.d_head), dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, 0.0)
        out = torch.einsum('b h l m, b m h d -> b l h d', attn, v)
        return self.out_proj(out.reshape(B, L, D))

    def _exp_map(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        c = torch.abs(self.curvature)
        factor = torch.tanh(c.sqrt() * norm) / (c.sqrt() * norm + 1e-8)
        return factor * x

    def _hyp_dist(self, x, y):
        num = 2 * torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=-1)
        denom = (1 - torch.sum(x**2, dim=-1).unsqueeze(2)) * (1 - torch.sum(y**2, dim=-1).unsqueeze(1))
        return torch.acosh(1 + num / (denom + 1e-8))

# ---------- FNO ----------
class FourierNeuralOperator(nn.Module):
    def __init__(self, d_model, n_modes=64):
        super().__init__()
        self.n_modes = n_modes
        self.R = nn.Parameter(torch.randn(d_model, d_model, n_modes, dtype=torch.cfloat) * 0.02)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        L_pad = 2 ** math.ceil(math.log2(L))
        x_padded = F.pad(x.transpose(1,2), (0, L_pad - L)).transpose(1,2)
        x_ft = torch.fft.rfft(x_padded, dim=1)
        x_ft_modes = x_ft[:, :self.n_modes, :]
        out_ft = torch.einsum('b m d, d o m -> b d o', x_ft_modes, self.R)
        out_ft_full = torch.zeros(B, x_ft.shape[1], D, dtype=torch.cfloat, device=x.device)
        out_ft_full[:, :self.n_modes, :] = out_ft
        out_padded = torch.fft.irfft(out_ft_full, dim=1)
        out = out_padded[:, :L, :]
        return self.norm(x + self.linear(out) + out)