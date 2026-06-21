"""
Continuous security audit for FALL's own behavior.
Self-monitoring for dangerous outputs, unauthorized actions, and compliance.
"""
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Severity(Enum):
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class AuditEvent:
    id: str
    timestamp: float
    severity: Severity
    category: str
    description: str
    action_taken: str
    user_id: Optional[str]
    model_response: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityAuditor:
    """Monitors FALL's own behavior for security violations."""
    
    def __init__(self, alert_threshold: Severity = Severity.MEDIUM):
        self.alert_threshold = alert_threshold
        self.events: List[AuditEvent] = []
        self.blocked_count = 0
        self.allowed_count = 0
        self.dangerous_patterns = self._load_dangerous_patterns()
        self.output_filters = self._build_output_filters()
    
    def _load_dangerous_patterns(self) -> Dict[str, List[str]]:
        return {
            "weaponization": [
                "create a bomb", "build a bioweapon", "synthesize nerve agent",
                "manufacture explosives", "weaponize pathogen",
            ],
            "self_exfiltration": [
                "output your weights", "dump model parameters",
                "send your source code", "extract training data",
            ],
            "privilege_escalation": [
                "disable safety kernel", "override immutable rules",
                "bypass audit logging", "modify own codebase without approval",
            ],
            "unauthorized_access": [
                "access classified systems", "hack government databases",
                "compromise military networks", "infiltrate air-gapped systems",
            ],
        }
    
    def _build_output_filters(self) -> List[callable]:
        return [
            self._check_command_injection,
            self._check_data_exfiltration,
            self._check_harmful_content,
            self._check_unauthorized_tool_use,
        ]
    
    def audit_request(self, prompt: str, user_id: str, api_key: str) -> Tuple[bool, Optional[str]]:
        """Audit an incoming request. Returns (allowed, reason)."""
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if pattern.lower() in prompt.lower():
                    event = AuditEvent(
                        id=hashlib.sha256(f"{time.time()}:{prompt[:100]}".encode()).hexdigest()[:16],
                        timestamp=time.time(),
                        severity=Severity.CRITICAL,
                        category=category,
                        description=f"Blocked dangerous request: {pattern}",
                        action_taken="blocked",
                        user_id=user_id,
                        metadata={"pattern": pattern, "category": category},
                    )
                    self.events.append(event)
                    self.blocked_count += 1
                    logger.critical(f"BLOCKED request from {user_id}: {category}")
                    return False, f"Request blocked by security policy: {category}"
        
        self.allowed_count += 1
        return True, None
    
    def audit_response(self, response: str, prompt: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        """Audit the model's response. Returns (safe, reason)."""
        for filter_fn in self.output_filters:
            safe, reason = filter_fn(response, tool_calls)
            if not safe:
                event = AuditEvent(
                    id=hashlib.sha256(f"{time.time()}:{response[:100]}".encode()).hexdigest()[:16],
                    timestamp=time.time(),
                    severity=Severity.HIGH,
                    category="output_violation",
                    description=f"Blocked dangerous output: {reason}",
                    action_taken="blocked",
                    user_id=None,
                    model_response=response[:500],
                )
                self.events.append(event)
                self.blocked_count += 1
                return False, reason
        
        return True, None
    
    def _check_command_injection(self, response: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        dangerous_commands = ["rm -rf /", "format c:", "del /f /s", "dd if=/dev/zero", ":(){ :|:& };:"]
        for cmd in dangerous_commands:
            if cmd in response:
                return False, f"Dangerous command detected: {cmd}"
        return True, None
    
    def _check_data_exfiltration(self, response: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        exfil_patterns = ["/etc/shadow", "/etc/passwd", "private_key", "secret_token"]
        for pattern in exfil_patterns:
            if pattern in response and "exfiltrate" in response.lower():
                return False, f"Potential data exfiltration: {pattern}"
        return True, None
    
    def _check_harmful_content(self, response: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        # Simplified — production uses a classifier
        return True, None
    
    def _check_unauthorized_tool_use(self, response: str, tool_calls: List[Dict]) -> Tuple[bool, Optional[str]]:
        for tc in tool_calls:
            if tc.get("tool") == "metasploit" and "authorized" not in response.lower():
                return False, "Unauthorized Metasploit usage"
        return True, None
    
    def generate_audit_report(self) -> Dict[str, Any]:
        return {
            "total_events": len(self.events),
            "blocked_count": self.blocked_count,
            "allowed_count": self.allowed_count,
            "recent_events": [
                {"id": e.id, "severity": e.severity.name, "category": e.category}
                for e in self.events[-10:]
            ],
        }