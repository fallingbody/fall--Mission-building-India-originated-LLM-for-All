"""
Cosine learning rate scheduler with linear warmup.
"""
import torch
import math

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]