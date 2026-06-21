"""
Event-driven architecture for FALL.
Enables decoupled communication between subsystems.
"""
import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Set, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    type: str
    data: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    id: Optional[str] = None

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 10000
        self.running = False
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.processed_count = 0
    
    def subscribe(self, event_type: str, handler: Callable[[Event], Awaitable[None]]):
        """Subscribe to an event type."""
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type."""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if not event.id:
            event.id = f"evt_{int(time.time()*1000)}_{len(self.event_history)}"
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Queue for async processing
        priority_value = -event.priority.value  # Negate for heapq (higher priority = lower value)
        await self.queue.put((priority_value, event))
    
    async def start_processing(self):
        """Start the event processing loop."""
        self.running = True
        
        while self.running:
            try:
                _, event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                handlers = self.subscribers.get(event.type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        await self.publish(Event(
                            type="system.error",
                            data={"error": str(e), "event": event.type, "handler": handler.__name__},
                            source="event_bus",
                            priority=EventPriority.HIGH,
                        ))
                
                self.processed_count += 1
            except asyncio.TimeoutError:
                pass
    
    async def stop(self):
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "processed_count": self.processed_count,
            "subscribers_count": sum(len(v) for v in self.subscribers.values()),
            "event_types": len(self.subscribers),
            "history_size": len(self.event_history),
        }


# Predefined FALL event types
class FALLEvents:
    # Training
    TRAINING_STEP_COMPLETE = "training.step_complete"
    CHECKPOINT_SAVED = "training.checkpoint_saved"
    TRAINING_ERROR = "training.error"
    
    # Inference
    INFERENCE_REQUEST = "inference.request"
    INFERENCE_COMPLETE = "inference.complete"
    INFERENCE_ERROR = "inference.error"
    
    # Agent
    AGENT_TASK_STARTED = "agent.task_started"
    AGENT_TASK_COMPLETE = "agent.task_complete"
    AGENT_TASK_FAILED = "agent.task_failed"
    AGENT_ANOMALY_DETECTED = "agent.anomaly_detected"
    
    # System
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_RESOURCE_ALERT = "system.resource_alert"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Security
    SECURITY_ALERT = "security.alert"
    SECURITY_AUDIT_EVENT = "security.audit_event"
    SECURITY_REDTEAM_FINDING = "security.redteam_finding"


# Global event bus instance
event_bus = EventBus()