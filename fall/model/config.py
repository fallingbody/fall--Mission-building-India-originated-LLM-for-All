from dataclasses import dataclass

@dataclass
class FALLConfig:
    vocab_size: int = 50257
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_head: int = 128
    d_ffn: int = 4096            # Expert hidden dim
    
    # MoE Configuration
    n_experts_per_layer: int = 4
    n_active_experts: int = 2
    kan_expert_ratio: float = 0.25
    
    # Domain Specialization
    expert_domains: tuple = ("general", "code", "math", "reasoning")
    use_thought_routing: bool = True
    
    # FNO Configuration
    fno_n_modes: int = 16         # Reduced from 64 to prevent GPU OOM at d_model=2048
    
    # Advanced Attention Configuration
    d_latent_kv: int = 128       # MLA compression
    csa_compression_ratio: int = 4 # Compressed Sparse Attention
    swa_window_size: int = 128   # Sliding Window Attention
    use_csa_attention: bool = True
    
    # Mamba
    d_state: int = 64            
    d_conv: int = 4              
    
    # Core settings
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    use_differential_attn: bool = True
    use_hyperbolic_layers: set = frozenset({6, 12, 18})
    use_fno_layers: set = frozenset({4, 14})
    use_mamba_layers: set = frozenset({3, 9, 15, 21})
    dropout: float = 0.1
    use_gradient_checkpointing: bool = True