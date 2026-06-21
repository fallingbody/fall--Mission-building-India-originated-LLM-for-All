"""
Real-time monitoring dashboard for FALL.
Tracks training, inference, agent state, and system health.
"""
import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import psutil
import torch

@dataclass
class SystemMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    disk_percent: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    timestamp: float = 0.0

@dataclass
class TrainingMetrics:
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    gradient_norm: float = 0.0
    expert_utilization: Dict[int, float] = field(default_factory=dict)
    router_entropy: float = 0.0
    timestamp: float = 0.0

@dataclass
class InferenceMetrics:
    total_requests: int = 0
    active_requests: int = 0
    avg_latency_ms: float = 0.0
    tokens_generated: int = 0
    requests_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    timestamp: float = 0.0

@dataclass
class AgentMetrics:
    uptime_seconds: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    mcts_simulations: int = 0
    prm_score_avg: float = 0.0
    motivation_score: float = 0.0
    timestamp: float = 0.0


class FALLMonitor:
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.system_history: deque = deque(maxlen=history_size)
        self.training_history: deque = deque(maxlen=history_size)
        self.inference_history: deque = deque(maxlen=history_size)
        self.agent_history: deque = deque(maxlen=history_size)
        self.alert_queue: deque = deque(maxlen=1000)

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            timestamp=time.time(),
        )

        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics.gpu_utilization.append(
                    torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0.0
                )
                mem = torch.cuda.memory_stats(i)
                metrics.gpu_memory.append(
                    mem.get("allocated_bytes.all.current", 0) / mem.get("reserved_bytes.all.current", 1) * 100
                )

        # Network
        net = psutil.net_io_counters()
        metrics.network_bytes_sent = net.bytes_sent
        metrics.network_bytes_recv = net.bytes_recv

        self.system_history.append(metrics)
        await self._check_alerts(metrics)
        return metrics

    async def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions."""
        if metrics.cpu_percent > 95:
            self.alert_queue.append({
                "level": "warning",
                "message": f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                "timestamp": time.time(),
            })
        if metrics.memory_percent > 95:
            self.alert_queue.append({
                "level": "critical",
                "message": f"Memory usage critical: {metrics.memory_percent:.1f}%",
                "timestamp": time.time(),
            })
        if any(g > 95 for g in metrics.gpu_utilization):
            self.alert_queue.append({
                "level": "warning",
                "message": f"GPU utilization critical",
                "timestamp": time.time(),
            })

    def record_training_step(self, metrics: TrainingMetrics):
        """Record a training step."""
        self.training_history.append(metrics)

    def record_inference(self, metrics: InferenceMetrics):
        """Record inference metrics."""
        self.inference_history.append(metrics)

    def record_agent_state(self, metrics: AgentMetrics):
        """Record agent state."""
        self.agent_history.append(metrics)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for the dashboard."""
        return {
            "system": self._latest_or_none(self.system_history),
            "training": self._latest_or_none(self.training_history),
            "inference": self._latest_or_none(self.inference_history),
            "agent": self._latest_or_none(self.agent_history),
            "alerts": list(self.alert_queue),
            "history": {
                "training_loss": [m.loss for m in self.training_history if m.loss > 0],
                "tokens_per_second": [m.tokens_per_second for m in self.training_history],
            },
        }

    def _latest_or_none(self, dq: deque) -> Optional[Any]:
        return dq[-1] if dq else None

    def generate_html_dashboard(self) -> str:
        """Generate an HTML dashboard."""
        data = self.get_dashboard_data()
        return f"""
<!DOCTYPE html>
<html>
<head><title>FALL Monitoring Dashboard</title>
<style>
    body {{ font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
    .card {{ border: 1px solid #00ff00; margin: 10px; padding: 15px; border-radius: 5px; }}
    .alert-warning {{ color: #ffaa00; }}
    .alert-critical {{ color: #ff0000; }}
    .metric {{ display: inline-block; margin: 10px; }}
    .value {{ font-size: 24px; font-weight: bold; }}
</style></head>
<body>
    <h1>⚡ FALL Monitoring Dashboard</h1>
    <div class="card">
        <h2>System</h2>
        <div class="metric">CPU: <span class="value">{data['system'].cpu_percent:.1f}%</span></div>
        <div class="metric">Memory: <span class="value">{data['system'].memory_percent:.1f}%</span></div>
    </div>
    <div class="card">
        <h2>Training</h2>
        <div class="metric">Step: <span class="value">{data['training'].step}</span></div>
        <div class="metric">Loss: <span class="value">{data['training'].loss:.4f}</span></div>
        <div class="metric">Tokens/s: <span class="value">{data['training'].tokens_per_second:,.0f}</span></div>
    </div>
    <div class="card">
        <h2>Agent</h2>
        <div class="metric">Uptime: <span class="value">{data['agent'].uptime_seconds/3600:.1f}h</span></div>
        <div class="metric">Tasks: <span class="value">{data['agent'].tasks_completed}</span></div>
    </div>
    <div class="card">
        <h2>Alerts</h2>
        {"".join(f'<div class="alert-{a["level"]}">{a["message"]}</div>' for a in data['alerts'][-10:])}
    </div>
</body>
</html>
"""