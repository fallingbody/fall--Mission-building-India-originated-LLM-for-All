from dataclasses import dataclass

@dataclass
class FALLConfig:
    vocab_size: int = 256_000
    d_model: int = 16384
    n_layers: int = 128
    n_heads: int = 128
    d_head: int = 128
    d_ffn: int = 65536           # Expert hidden dim
    n_experts_per_layer: int = 64
    n_active_experts: int = 2
    d_latent_kv: int = 2048      # MLA compression
    d_state: int = 128           # Mamba state
    d_conv: int = 4              # Mamba conv width
    max_seq_len: int = 524288
    rope_base: float = 50_000_000.0
    use_differential_attn: bool = True
    use_hyperbolic_layers: set = frozenset({4, 8, 16, 32, 64})
    use_fno_layers: set = frozenset({8, 16, 24, 32, 40, 48, 56, 64})
    use_mamba_layers: set = frozenset({16, 32, 48, 64, 80, 96, 112, 128})
    kan_expert_ratio: float = 0.2  # 20% KAN experts
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True