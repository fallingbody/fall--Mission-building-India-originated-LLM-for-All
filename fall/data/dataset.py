"""
Streaming dataset for FALL training.
Provides infinite stream of packed sequences.
"""
import torch
from torch.utils.data import IterableDataset
import numpy as np
from typing import Iterator, Dict

class FALLDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_dir: str,
        seq_len: int = 524288,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed sequences."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        buffer = []
        for file_path in self._get_file_list():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip empty
                    if not line.strip():
                        continue
                    # Tokenize
                    tokens = self.tokenizer.encode(line)
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

    def _get_file_list(self):
        import glob
        files = sorted(glob.glob(f"{self.data_dir}/**/*.txt", recursive=True))
        return files[self.rank::self.world_size]