"""
Main FALL training loop.
Supports FSDP, FP8, gradient checkpointing, and async checkpointing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
import time
import os

from fall.model.model import FALLForCausalLM
from fall.model.config import FALLConfig
from fall.training.fsdp_wrap import apply_fsdp
from fall.training.optimizer import create_optimizer
from fall.training.scheduler import CosineWarmupScheduler
from fall.training.fp8_utils import get_fp8_context
from fall.training.checkpoint import AsyncCheckpointer

class FALLTrainer:
    def __init__(
        self,
        config: FALLConfig,
        train_dataloader,
        val_dataloader=None,
        checkpoint_dir="./checkpoints",
        total_steps=3_750_000,
        warmup_steps=5_000,
        save_every=1_000,
        log_every=100,
        resume=False,
    ):
        self.config = config
        if train_dataloader is None:
            raise ValueError("train_dataloader must be provided! Dummy data is no longer accepted.")
        self.train_dataloader = train_dataloader
            
        self.val_dataloader = val_dataloader
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.save_every = save_every
        self.log_every = log_every

        self.device_id = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")
        self.model = FALLForCausalLM(config).to(self.device)
        if dist.is_initialized():
            self.model = apply_fsdp(self.model, self.device_id)

        # Optimizer
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps
        )

        # Checkpointer
        self.checkpointer = AsyncCheckpointer(
            self.model, self.optimizer, checkpoint_dir, save_every, self.world_size
        )

        # Mixed precision fallback for Colab
        self.fp8_context = get_fp8_context(enabled=True)
        if type(self.fp8_context).__name__ == "nullcontext":
            self.fp8_context = torch.autocast(device_type="cuda", dtype=torch.float16)

        # Gradient accumulation
        self.grad_accum_steps = 1  # Set based on micro-batch size

        # Resume from checkpoint
        self.current_step = 0
        if resume:
            self.current_step = self._resume()

    def _resume(self):
        """Try to resume from latest checkpoint."""
        import glob
        checkpoints = glob.glob(os.path.join(self.checkpointer.save_dir, "step_*.pt"))
        if not checkpoints:
            print("No checkpoints found to resume from.")
            return 0
        
        # Find highest step
        latest_ckpt = max(checkpoints, key=lambda x: int(os.path.basename(x).replace("step_", "").replace(".pt", "")))
        print(f"Resuming from checkpoint: {latest_ckpt}")
        
        state = torch.load(latest_ckpt, map_location=self.device)
        self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        else:
            print("No optimizer state found in checkpoint. Starting with fresh optimizer.")
        return state["step"] + 1

    def train(self):
        """Main training loop."""
        self.model.train()
        data_iter = iter(self.train_dataloader)
        total_loss = 0.0

        for step in range(self.current_step, self.total_steps):
            t_start = time.time()

            # Load batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with FP8
            with self.fp8_context:
                logits = self.model(input_ids)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Step
            self.optimizer.step()
            self.scheduler.step(step)
            self.optimizer.zero_grad()

            total_loss += loss.item()

            # Logging
            if step % self.log_every == 0 and self.rank == 0:
                avg_loss = total_loss / self.log_every
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                tokens_per_sec = (input_ids.numel() * self.world_size) / elapsed
                print(f"Step {step:>7d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/s: {tokens_per_sec:,.0f}")
                total_loss = 0.0

            # Checkpoint
            if step % self.save_every == 0:
                self.checkpointer.save(step)
                if self.rank == 0:
                    print(f"Checkpoint saved at step {step}")

        # Final save
        self.checkpointer.save(self.total_steps)

    def validate(self):
        """Validation loop."""
        if self.val_dataloader is None:
            return
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device_id)
                labels = batch["labels"].to(self.device_id)
                logits = self.model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        if self.rank == 0:
            print(f"Validation | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f}")
        self.model.train()


def launch_training(config, train_dataloader, val_dataloader=None):
    """Entry point for distributed training."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    trainer = FALLTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.train()
    dist.destroy_process_group()