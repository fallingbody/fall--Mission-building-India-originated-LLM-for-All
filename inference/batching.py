"""
Dynamic batching scheduler for FALL inference.
Maximizes throughput while maintaining latency SLOs.
"""
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq

@dataclass(order=True)
class InferenceRequest:
    priority: int
    arrival_time: float
    sequence: torch.Tensor
    max_tokens: int
    temperature: float
    request_id: str
    future: asyncio.Future = field(compare=False)

class DynamicBatcher:
    def __init__(
        self,
        max_batch_size: int = 256,
        max_wait_ms: float = 50.0,
        target_tokens_per_batch: int = 16384,
        max_seq_len: int = 524288,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.target_tokens_per_batch = target_tokens_per_batch
        self.max_seq_len = max_seq_len
        self.pending: List[InferenceRequest] = []
        self.active_batches: List[List[InferenceRequest]] = []
    
    async def schedule(self, request: InferenceRequest) -> asyncio.Future:
        """Schedule an inference request, returning a future."""
        heapq.heappush(self.pending, request)
        return request.future
    
    async def build_batch(self) -> Optional[List[InferenceRequest]]:
        """Build an optimal batch from pending requests."""
        if not self.pending:
            return None
        
        batch = []
        total_tokens = 0
        now = time.time()
        
        while self.pending and len(batch) < self.max_batch_size:
            # Check if oldest request has waited too long
            oldest = self.pending[0]
            wait_time = now - oldest.arrival_time
            
            if wait_time > self.max_wait_ms or len(batch) == 0:
                request = heapq.heappop(self.pending)
                batch.append(request)
                total_tokens += min(request.sequence.shape[1], self.max_seq_len)
                
                # Check if we have enough tokens
                if total_tokens >= self.target_tokens_per_batch:
                    break
            else:
                break
        
        return batch if batch else None
    
    async def process_batch(
        self,
        model: torch.nn.Module,
        batch: List[InferenceRequest],
    ) -> None:
        """Process a batch of requests through the model."""
        # Pad sequences to max length in batch
        max_len = max(req.sequence.shape[1] for req in batch)
        
        padded = []
        for req in batch:
            seq = req.sequence
            if seq.shape[1] < max_len:
                pad = torch.zeros(1, max_len - seq.shape[1], dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=1)
            padded.append(seq)
        
        inputs = torch.cat(padded, dim=0)
        
        # Run model
        with torch.no_grad():
            logits = model(inputs)
        
        # Distribute results
        for i, req in enumerate(batch):
            req_logits = logits[i:i+1, :req.sequence.shape[1], :]
            req.future.set_result(req_logits)