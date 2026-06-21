"""
μTransfer optimizer for FALL.
Learning rates are scaled by hidden dimension width.
"""
import torch
import torch.nn as nn

def create_optimizer(model, config):
    """
    Create AdamW optimizer with μTransfer scaling.
    LR scales as 1/d_model for input projections, 1/sqrt(d_model) for output.
    """
    # Separate parameters by type for μTransfer
    input_params = []   # Embedding, Q/K/V input projections
    output_params = []  # Output projections, LM head
    other_params = []   # Layer norms, biases, router

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ['embed', 'q_proj', 'k_proj', 'v_proj',
                                      'in_proj', 'kv_compress', 'gate']):
            input_params.append(param)
        elif any(x in name for x in ['out_proj', 'lm_head', 'output']):
            output_params.append(param)
        else:
            other_params.append(param)

    # μTransfer scaling
    d_model = config.d_model
    param_groups = [
        {'params': input_params,  'lr': 1.5e-4 / d_model},
        {'params': output_params, 'lr': 1.5e-4 / (d_model ** 0.5)},
        {'params': other_params,  'lr': 1.5e-4},
    ]

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=False  # Disabled because FNO uses complex64 which isn't supported by fused kernel
    )