"""
Emergency shutdown system for FALL.
Provides hardware-level kill switch capability.
"""
import asyncio
import os
import signal
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ShutdownLevel(Enum):
    SOFT = 1       # Stop accepting new requests
    HARD = 2       # Terminate all active tasks
    EMERGENCY = 3  # Immediate power-off signal


@dataclass
class ShutdownEvent:
    level: ShutdownLevel
    reason: str
    timestamp: float
    triggered_by: str


class EmergencyShutdown:
    def __init__(self):
        self.active = False
        self.shutdown_level: Optional[ShutdownLevel] = None
        self.shutdown_history: list = []
        self.on_shutdown_callbacks: list = []
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register OS signal handlers for immediate shutdown."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigint)
    
    def _handle_sigterm(self, signum, frame):
        self.shutdown(ShutdownLevel.HARD, "SIGTERM received")
    
    def _handle_sigint(self, signum, frame):
        self.shutdown(ShutdownLevel.SOFT, "SIGINT received")
    
    def register_callback(self, callback: Callable):
        self.on_shutdown_callbacks.append(callback)
    
    def shutdown(self, level: ShutdownLevel, reason: str, triggered_by: str = "system"):
        if self.active and self.shutdown_level == ShutdownLevel.EMERGENCY:
            return  # Already in emergency shutdown
        
        event = ShutdownEvent(
            level=level,
            reason=reason,
            timestamp=time.time(),
            triggered_by=triggered_by,
        )
        self.shutdown_history.append(event)
        self.active = True
        self.shutdown_level = level
        
        print(f"\n{'='*60}")
        print(f"SHUTDOWN INITIATED: {level.name}")
        print(f"Reason: {reason}")
        print(f"{'='*60}\n")
        
        for callback in self.on_shutdown_callbacks:
            try:
                callback(level, reason)
            except Exception as e:
                print(f"Shutdown callback failed: {e}")
        
        if level == ShutdownLevel.EMERGENCY:
            self._emergency_halt()
    
    def _emergency_halt(self):
        """Immediate hardware-level shutdown."""
        try:
            # Attempt to sync filesystems
            os.sync()
        except Exception:
            pass
        
        # Send power-off command (platform-dependent)
        try:
            if os.name == "posix":
                os.system("poweroff -f")
            else:
                os._exit(1)
        except Exception:
            os._exit(1)
    
    def verify_safety(self) -> bool:
        """Verify all safety systems are operational."""
        checks = [
            self._check_audit_logging(),
            self._check_kill_switch_functional(),
        ]
        return all(checks)
    
    def _check_audit_logging(self) -> bool:
        return True  # Placeholder
    
    def _check_kill_switch_functional(self) -> bool:
        return True  # Placeholder


# Global instance
emergency = EmergencyShutdown()