"""
Training launch script for FALL.
Supports multi-node, multi-GPU via torchrun.
"""
import torch
import torch.distributed as dist
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fall_5t.yaml")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fp8", action="store_true", default=True)
    args = parser.parse_args()

    # Load config
    from fall.model.config import FALLConfig
    config = FALLConfig()

    # Setup distributed
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # Create dataloaders (placeholder — real implementation in data/ module)
    train_dataloader = None  # Will be replaced by actual data pipeline
    val_dataloader = None

    # Launch training
    from fall.training.trainer import FALLTrainer
    trainer = FALLTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()