"""
Fully Sharded Data Parallel wrapping policy for FALL.
Handles 5D parallelism: TP, PP, EP, DP, SP.
"""
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    _or_policy,
)
import functools
from fall.model.layers import FALLDecoderLayer
from fall.model.moe import AuxiliaryLossFreeMoE

def get_fsdp_wrap_policy():
    """Wrap each decoder layer and each MoE block separately."""
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            FALLDecoderLayer,
            AuxiliaryLossFreeMoE,
        }
    )

def apply_fsdp(model, device_id, mixed_precision=None):
    """Apply FSDP with ZeRO-3 sharding to the model."""
    return FSDP(
        model,
        auto_wrap_policy=get_fsdp_wrap_policy(),
        device_id=device_id,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        cpu_offload=None,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,
        sync_module_states=True,
        param_init_fn=None,
    )