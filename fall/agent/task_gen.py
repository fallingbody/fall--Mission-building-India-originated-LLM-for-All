"""
Autonomous task generator for FALL.
Creates the agent's daily agenda without human input.
"""
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TaskTemplate:
    domain: str
    description: str
    priority_base: float
    estimated_duration: int

class AutonomousTaskGenerator:
    def __init__(self):
        self.templates = self._load_templates()
        self.history = []
        self.current_agenda = []

    def _load_templates(self) -> List[TaskTemplate]:
        return [
            # Security
            TaskTemplate("security", "Scan network for new devices", 0.8, 300),
            TaskTemplate("security", "Audit running services for CVEs", 0.9, 600),
            TaskTemplate("security", "Analyze auth logs for brute force", 0.7, 120),
            TaskTemplate("security", "Check firewall rules", 0.6, 60),
            TaskTemplate("security", "Verify TLS certificate expiry", 0.5, 30),
            # System
            TaskTemplate("system", "Clean up orphan processes", 0.4, 60),
            TaskTemplate("system", "Check disk usage", 0.3, 30),
            TaskTemplate("system", "Rotate logs", 0.3, 30),
            TaskTemplate("system", "Update package indices", 0.5, 120),
            # Knowledge
            TaskTemplate("knowledge", "Read latest CVEs", 0.8, 300),
            TaskTemplate("knowledge", "Study new exploit techniques", 0.7, 600),
            TaskTemplate("knowledge", "Analyze trending malware", 0.6, 600),
            # Code
            TaskTemplate("code", "Review recent commits for bugs", 0.6, 300),
            TaskTemplate("code", "Write missing tests", 0.4, 600),
            TaskTemplate("code", "Refactor complex functions", 0.3, 600),
            # Self-improvement
            TaskTemplate("self", "Analyze own error logs", 0.7, 120),
            TaskTemplate("self", "Fine-tune on recent experiences", 0.6, 600),
            TaskTemplate("self", "Optimize internal strategies", 0.5, 300),
        ]

    def generate_agenda(self, world_state: Dict) -> List[str]:
        """Generate daily agenda based on world state."""
        agenda = []

        # Always check security
        risk_level = world_state.get("threat_level", 0.0)
        if risk_level > 0.5:
            agenda.extend(self._get_templates("security", count=3))

        # Rotate through domains
        for domain in ["security", "knowledge", "code", "system", "self"]:
            tasks = self._get_templates(domain, count=1)
            agenda.extend(tasks)

        # Prioritize
        agenda = self._prioritize(agenda, world_state)
        self.current_agenda = agenda
        return agenda

    def _get_templates(self, domain: str, count: int = 1) -> List[str]:
        matching = [t for t in self.templates if t.domain == domain]
        selected = random.sample(matching, min(count, len(matching)))
        return [t.description for t in selected]

    def _prioritize(self, tasks: List[str], world_state: Dict) -> List[str]:
        scored = [(t, self._score_task(t, world_state)) for t in tasks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in scored]

    def _score_task(self, task: str, world_state: Dict) -> float:
        for template in self.templates:
            if template.description == task:
                return template.priority_base
        return 0.5