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
        if dist.is_initialized():
            # Use FSDP context to rebuild the scattered shards into one CPU tensor
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp import StateDictType
            
            if isinstance(self.model, FSDP):
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state = self.model.state_dict()
            else:
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                
            if dist.get_rank() == 0:
                state = {
                    "model": model_state,
                    "step": step,
                }
                checkpoint_path = os.path.join(self.save_dir, f"step_{step}.pt")
                torch.save(state, checkpoint_path)
            # DDP requires a barrier so other ranks wait until save is done
            dist.barrier()
        else:
            state = {
                "model": self.model.state_dict(),
                "step": step,
            }
            checkpoint_path = os.path.join(self.save_dir, f"step_{step}.pt")
            torch.save(state, checkpoint_path)
            
        self._cleanup_old_checkpoints(step)

    def _cleanup_old_checkpoints(self, current_step):
        import glob
        import shutil
        
        # Cleanup single-file checkpoints
        for ckpt in glob.glob(os.path.join(self.save_dir, "step_*.pt")):
            try:
                ckpt_step = int(os.path.basename(ckpt).replace("step_", "").replace(".pt", ""))
                if ckpt_step < current_step:
                    os.remove(ckpt)
            except ValueError:
                pass
                
        # Cleanup distributed checkpoint directories
        for ckpt_dir in glob.glob(os.path.join(self.save_dir, "step_*")):
            if os.path.isdir(ckpt_dir):
                try:
                    ckpt_step = int(os.path.basename(ckpt_dir).replace("step_", ""))
                    if ckpt_step < current_step:
                        shutil.rmtree(ckpt_dir)
                except ValueError:
                    pass

    def _get_storage(self):
        from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
        return FileSystemWriter(self.save_dir)