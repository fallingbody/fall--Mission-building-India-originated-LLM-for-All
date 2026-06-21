"""
Prometheus metrics for FALL.
"""
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time

# Training metrics
training_loss = Gauge("fall_training_loss", "Current training loss")
training_lr = Gauge("fall_training_learning_rate", "Current learning rate")
training_steps_total = Counter("fall_training_steps_total", "Total training steps")
training_tokens_total = Counter("fall_training_tokens_total", "Total tokens processed")

# Inference metrics
inference_requests_total = Counter("fall_inference_requests_total", "Total inference requests")
inference_latency_seconds = Histogram(
    "fall_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 60.0],
)
inference_tokens_generated = Counter("fall_inference_tokens_generated", "Total tokens generated")

# Agent metrics
agent_tasks_completed = Counter("fall_agent_tasks_completed", "Total autonomous tasks completed")
agent_uptime_seconds = Gauge("fall_agent_uptime_seconds", "Agent uptime in seconds")
agent_current_task = Gauge("fall_agent_current_task", "Current task ID")

# System metrics
gpu_utilization = Gauge("fall_gpu_utilization", "GPU utilization percent", ["device"])
gpu_memory_used = Gauge("fall_gpu_memory_used_bytes", "GPU memory used", ["device"])
cpu_utilization = Gauge("fall_cpu_utilization", "CPU utilization percent")
memory_used = Gauge("fall_memory_used_bytes", "Memory used")

def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")