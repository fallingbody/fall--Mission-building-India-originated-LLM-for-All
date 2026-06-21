"""
FP8 mixed precision utilities using Transformer Engine.
"""
import torch

try:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.common.recipe import Format, FP8Format
    HAS_TE = True
except ImportError:
    HAS_TE = False
    fp8_autocast = None

def get_fp8_context(enabled=True):
    """Return FP8 autocast context if available, else nullcontext."""
    if HAS_TE and enabled:
        from transformer_engine.common.recipe import DelayedScaling
        return fp8_autocast(enabled=True, fp8_recipe=DelayedScaling())
    else:
        import contextlib
        return contextlib.nullcontext()

def apply_fp8_to_model(model):
    """Wrap model layers with FP8 using Transformer Engine."""
    if not HAS_TE:
        return model
    # TE would replace nn.Linear with te.Linear; simplified here
    return model