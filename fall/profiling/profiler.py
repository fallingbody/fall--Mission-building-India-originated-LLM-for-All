"""
Performance profiling for FALL.
Tracks latency, memory, throughput across all components.
"""
import torch
import time
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    name: str
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    calls: int = 0
    tokens_processed: int = 0
    memory_allocated_mb: float = 0.0
    memory_peak_mb: float = 0.0

class FALLProfiler:
    def __init__(self):
        self.results: Dict[str, ProfileResult] = {}
        self.enabled = True
    
    @contextmanager
    def profile(self, name: str):
        """Context manager to profile a block."""
        if not self.enabled:
            yield
            return
        
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_start = torch.cuda.memory_allocated()
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_end = torch.cuda.memory_allocated()
            mem_delta = (mem_end - mem_start) / (1024 * 1024)
        else:
            mem_delta = 0.0
        
        duration = (time.time() - start) * 1000
        
        if name not in self.results:
            self.results[name] = ProfileResult(name=name)
        
        r = self.results[name]
        r.total_time_ms += duration
        r.min_time_ms = min(r.min_time_ms, duration)
        r.max_time_ms = max(r.max_time_ms, duration)
        r.calls += 1
        r.memory_allocated_mb += mem_delta
    
    def profile_function(self, func: Callable):
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile(func.__qualname__):
                return func(*args, **kwargs)
        return wrapper
    
    def record_tokens(self, name: str, count: int):
        if name in self.results:
            self.results[name].tokens_processed += count
    
    def get_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 80, "FALL Performance Profile", "=" * 80]
        lines.append(f"{'Component':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        lines.append("-" * 80)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1].total_time_ms, reverse=True):
            avg = result.total_time_ms / max(1, result.calls)
            lines.append(
                f"{name:<40} {result.calls:>8} {result.total_time_ms/1000:>10.2f} "
                f"{avg:>10.1f} {result.min_time_ms:>10.1f} {result.max_time_ms:>10.1f}"
            )
        
        lines.append("-" * 80)
        total_time = sum(r.total_time_ms for r in self.results.values()) / 1000
        lines.append(f"Total profiled time: {total_time:.2f}s")
        return "\n".join(lines)
    
    def get_json(self) -> Dict:
        return {
            name: {
                "total_time_ms": r.total_time_ms,
                "min_time_ms": r.min_time_ms,
                "max_time_ms": r.max_time_ms,
                "calls": r.calls,
                "tokens_processed": r.tokens_processed,
                "throughput_tokens_per_sec": r.tokens_processed / max(0.001, r.total_time_ms / 1000),
            }
            for name, r in self.results.items()
        }
    
    def reset(self):
        self.results.clear()


# Global profiler instance
profiler = FALLProfiler()