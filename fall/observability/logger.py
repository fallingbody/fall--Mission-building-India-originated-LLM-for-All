"""
Structured logging for FALL with OpenTelemetry integration.
"""
import logging
import json
import time
import sys
from typing import Optional, Dict, Any
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

tracer = trace.get_tracer("fall")
meter = metrics.get_meter("fall")

# Metrics
tokens_generated = meter.create_counter(
    "fall.tokens.generated",
    description="Total tokens generated",
)

inference_latency = meter.create_histogram(
    "fall.inference.latency",
    description="Inference latency in milliseconds",
    unit="ms",
)

training_steps = meter.create_counter(
    "fall.training.steps",
    description="Training steps completed",
)

class StructuredLogger:
    def __init__(self, name: str = "fall"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(self._format(msg, kwargs))
    
    def warn(self, msg: str, **kwargs):
        self.logger.warning(self._format(msg, kwargs))
    
    def error(self, msg: str, **kwargs):
        self.logger.error(self._format(msg, kwargs))
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(self._format(msg, kwargs))
    
    def _format(self, msg: str, extra: Dict) -> str:
        if extra:
            return f"{msg} | {json.dumps(extra, default=str)}"
        return msg

log = StructuredLogger()