"""
Asynchronous distributed checkpointing for FALL.
"""
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import asyncio
import os

class AsyncCheckpointer:
    def __init__(self, model, optimizer, save_dir, save_every=1000, world_size=1):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_every = save_every
        self.world_size = world_size
        os.makedirs(save_dir, exist_ok=True)

    def save(self, step):
        """Synchronous save (for critical checkpoints)."""
        state = {
            "model": get_model_state_dict(self.model),
            "optimizer": get_optimizer_state_dict(self.model, self.optimizer),
            "step": torch.tensor(step),
        }
        dcp.save(state, checkpoint_id=f"step_{step}", storage_writer=self._get_storage())

    def _get_storage(self):
        from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
        return FileSystemWriter(self.save_dir)