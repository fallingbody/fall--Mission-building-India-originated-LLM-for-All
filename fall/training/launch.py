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
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--fsdp", action="store_true", help="Enable Fully Sharded Data Parallel for training large models (1B+ parameters).")
    args = parser.parse_args()

    # Load config
    from fall.model.config import FALLConfig
    config = FALLConfig()

    config.max_seq_len = args.seq_len

    # Setup distributed
    if "LOCAL_RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1

    # Create dataloaders
    from transformers import AutoTokenizer
    from fall.data.dataset import FALLDataset
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    train_dataset = FALLDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        seq_len=config.max_seq_len,
        batch_size=1,
        rank=global_rank,
        world_size=world_size
    )
    
    # batch_size=1 to minimize activation memory for large models
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = None

    # Launch training
    from fall.training.trainer import FALLTrainer
    trainer = FALLTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()