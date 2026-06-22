"""
Streaming dataset for FALL training using HuggingFace Datasets.
Provides infinite stream of packed sequences.
"""
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Iterator, Dict

class FALLDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        seq_len: int = 256,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Initialize the streaming dataset
        self.dataset = load_dataset(
            self.dataset_name, 
            name=self.dataset_config if self.dataset_config else None, 
            split="train", 
            streaming=True
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed sequences."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
            
        total_shards = self.world_size * num_workers
        global_shard_id = self.rank * num_workers + worker_id

        # Try to shard the dataset if supported
        try:
            sharded_dataset = self.dataset.shard(num_shards=total_shards, index=global_shard_id)
        except Exception:
            sharded_dataset = self.dataset

        buffer = []
        for item in sharded_dataset:
            # Get text from the dataset (handles different column names like 'text' or 'content')
            text = item.get('text', item.get('content', ''))
            if not text.strip():
                continue
                
            # Tokenize
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 10:
                continue
            buffer.extend(tokens)
            
            # Yield packed sequences
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}