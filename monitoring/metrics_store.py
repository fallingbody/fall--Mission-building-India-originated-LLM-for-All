"""
Time-series metrics store for FALL dashboard.
"""
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import threading


class MetricsStore:
    """In-memory time-series metrics with automatic downsampling."""
    
    def __init__(self, retention_seconds: int = 86400):
        self.retention = retention_seconds
        self.metrics: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        # Pre-allocate common metrics
        self._init_metric("training_loss", maxlen=100000)
        self._init_metric("tokens_per_second", maxlen=100000)
        self._init_metric("gpu_utilization", maxlen=100000)
        self._init_metric("inference_latency_ms", maxlen=100000)
        self._init_metric("agent_tasks_completed", maxlen=10000)
    
    def _init_metric(self, name: str, maxlen: int = 100000):
        self.metrics[name] = deque(maxlen=maxlen)
    
    def record(self, name: str, value: float, timestamp: Optional[float] = None):
        with self.lock:
            if name not in self.metrics:
                self._init_metric(name)
            ts = timestamp or time.time()
            self.metrics[name].append((ts, value))
    
    def query(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        downsample: int = 1,
    ) -> List[Tuple[float, float]]:
        """Query a metric with optional time range and downsampling."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            data = list(self.metrics[name])
        
        if start_time:
            data = [(t, v) for t, v in data if t >= start_time]
        if end_time:
            data = [(t, v) for t, v in data if t <= end_time]
        
        if downsample > 1:
            downsampled = []
            for i in range(0, len(data), downsample):
                chunk = data[i:i+downsample]
                avg_time = sum(t for t, _ in chunk) / len(chunk)
                avg_value = sum(v for _, v in chunk) / len(chunk)
                downsampled.append((avg_time, avg_value))
            data = downsampled
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                name: {
                    "count": len(deque),
                    "latest": deque[-1] if deque else None,
                    "avg_last_100": (
                        sum(v for _, v in list(deque)[-100:]) / min(len(deque), 100)
                        if deque else 0
                    ),
                }
                for name, deque in self.metrics.items()
            }