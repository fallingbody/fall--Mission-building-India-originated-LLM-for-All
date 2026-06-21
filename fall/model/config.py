from dataclasses import dataclass

@dataclass
class FALLConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    d_ffn: int = 3072            # Expert hidden dim
    n_experts_per_layer: int = 8
    n_active_experts: int = 2
    d_latent_kv: int = 128       # MLA compression
    d_state: int = 64            # Mamba state
    d_conv: int = 4              # Mamba conv width
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    use_differential_attn: bool = True
    use_hyperbolic_layers: set = frozenset({4, 8})
    use_fno_layers: set = frozenset({2, 6, 10})
    use_mamba_layers: set = frozenset({3, 7, 11})
    kan_expert_ratio: float = 0.2  # 20% KAN experts
    dropout: float = 0.1
    use_gradient_checkpointing: bool = True