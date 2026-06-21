"""
Lightweight dependency injection container for FALL.
Manages component lifecycle and wiring.
"""
from typing import Dict, Any, Callable, Optional, Type, TypeVar
from dataclasses import dataclass
import inspect

T = TypeVar('T')

@dataclass
class ServiceDefinition:
    factory: Callable[..., Any]
    singleton: bool = True
    instance: Optional[Any] = None
    dependencies: list = None

class Container:
    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        singleton: bool = True,
    ):
        """Register a service factory."""
        self._services[name] = ServiceDefinition(
            factory=factory,
            singleton=singleton,
        )
    
    def register_instance(self, name: str, instance: Any):
        """Register an existing instance."""
        self._instances[name] = instance
        self._services[name] = ServiceDefinition(
            factory=lambda: instance,
            singleton=True,
            instance=instance,
        )
    
    def resolve(self, name: str) -> Any:
        """Resolve a service by name."""
        if name in self._instances:
            return self._instances[name]
        
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        
        service = self._services[name]
        instance = service.factory()
        
        if service.singleton:
            self._instances[name] = instance
        
        return instance
    
    def resolve_all(self, *names: str) -> tuple:
        """Resolve multiple services at once."""
        return tuple(self.resolve(name) for name in names)


# Global container
container = Container()

# Register core services
def register_core_services():
    from fall.model.config import FALLConfig
    from fall.inference.server import FALLInferenceServer, InferenceConfig
    from fall.sandbox.manager import SandboxManager, SandboxConfig
    from fall.agent.runtime import FALLAutonomousAgent
    from fall.security.audit import SecurityAuditor
    from fall.events.bus import EventBus, register_all_handlers
    
    # Config
    container.register_instance("config", FALLConfig())
    container.register_instance("inference_config", InferenceConfig())
    container.register_instance("sandbox_config", SandboxConfig())
    
    # Event bus
    bus = EventBus()
    container.register_instance("event_bus", bus)
    
    # Core components
    container.register("model", lambda: FALLForCausalLM(container.resolve("config")))
    container.register("inference_server", lambda: FALLInferenceServer(container.resolve("inference_config")))
    container.register("sandbox", lambda: SandboxManager(container.resolve("sandbox_config")))
    container.register("auditor", lambda: SecurityAuditor())
    container.register(
        "agent",
        lambda: FALLAutonomousAgent(
            model_server=container.resolve("inference_server"),
            sandbox=container.resolve("sandbox"),
        ),
    )