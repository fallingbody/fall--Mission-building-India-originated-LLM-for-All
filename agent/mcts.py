"""
Monte Carlo Tree Search planner for FALL.
Explores action sequences, guided by PRM and value function.
"""
import math
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCTSNode:
    state: Any
    action: Optional[str] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False

    def ucb_score(self, c_puct: float = 1.5) -> float:
        if self.visits == 0:
            return float('inf')
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value / self.visits + exploration

    def add_child(self, child: 'MCTSNode'):
        self.children.append(child)
        child.parent = self

class MCTSPlanner:
    def __init__(
        self,
        model_server,
        sandbox,
        num_simulations: int = 100,
        max_depth: int = 20,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        self.model = model_server
        self.sandbox = sandbox
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.action_space = self._define_actions()

    def _define_actions(self) -> Dict[str, Dict]:
        return {
            "recon": {
                "description": "Gather information about the target",
                "tools": ["nmap", "dnsrecon", "whois", "curl"],
            },
            "exploit": {
                "description": "Execute an exploit",
                "tools": ["metasploit", "python", "shell"],
            },
            "escalate": {
                "description": "Privilege escalation",
                "tools": ["shell", "python"],
            },
            "exfiltrate": {
                "description": "Extract data",
                "tools": ["shell", "curl", "python"],
            },
            "persist": {
                "description": "Install persistence",
                "tools": ["shell", "cron", "registry"],
            },
            "clean": {
                "description": "Clear logs and traces",
                "tools": ["shell"],
            },
            "analyze": {
                "description": "Analyze results and plan next step",
                "tools": ["python", "file_read"],
            },
            "wait": {
                "description": "Wait and observe",
                "tools": [],
            },
            "abort": {
                "description": "Abort the mission",
                "tools": [],
            },
        }

    async def plan(self, task: str, world_state: Dict) -> 'Plan':
        """Create a plan for the given task using MCTS."""
        root = MCTSNode(state={"task": task, "world": world_state})
        start_time = time.time()

        for sim in range(self.num_simulations):
            node = root
            path = [node]
            depth = 0

            # Selection
            while node.children and not node.is_terminal and depth < self.max_depth:
                node = self._select_child(node)
                path.append(node)
                depth += 1

            # Expansion
            if not node.is_terminal and depth < self.max_depth:
                node = await self._expand(node, task)

            # Simulation
            if not node.is_terminal:
                value = await self._simulate(node, task, depth)
            else:
                value = node.value

            # Backpropagation
            self._backpropagate(path, value)

        logger.info(f"MCTS planning completed in {time.time() - start_time:.2f}s with {self.num_simulations} simulations")
        return self._extract_plan(root)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB."""
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    async def _expand(self, node: MCTSNode, task: str) -> MCTSNode:
        """Expand a node by adding possible actions."""
        for action_name, action_def in self.action_space.items():
            child = MCTSNode(
                state={"action": action_name, "tools": action_def["tools"]},
                action=action_name,
                prior=1.0 / len(self.action_space),
            )
            node.add_child(child)
        if node.children:
            return random.choice(node.children)
        return node

    async def _simulate(self, node: MCTSNode, task: str, depth: int) -> float:
        """Simulate from this node to estimate value."""
        # Use the model to predict the outcome
        prompt = self._build_simulation_prompt(node, task, depth)
        try:
            response = await self.model.generate(prompt, max_tokens=256)
            # Parse estimated success probability
            value = self._parse_value(response.get("text", ""))
        except Exception:
            value = 0.0
        return value

    def _build_simulation_prompt(self, node: MCTSNode, task: str, depth: int) -> str:
        return f"""<|think|>
Task: {task}
Current depth: {depth}
Action taken: {node.action}
Remaining depth: {self.max_depth - depth}

Estimate the probability of success (0.0 to 1.0) for this path.
Output only the number.
</|think|>
"""

    def _parse_value(self, text: str) -> float:
        try:
            text = text.strip()
            if text:
                return float(text.split()[0])
        except (ValueError, IndexError):
            pass
        return 0.0

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value up the tree."""
        for node in reversed(path):
            node.visits += 1
            node.value += value

    def _extract_plan(self, root: MCTSNode) -> 'Plan':
        """Extract the best plan from the MCTS tree."""
        steps = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            if node.action:
                steps.append(PlanStep(
                    action=node.action,
                    tools=node.state.get("tools", []),
                    confidence=node.value / max(1, node.visits),
                ))
        return Plan(steps=steps, root_node=root)


@dataclass
class PlanStep:
    action: str
    tools: List[str]
    confidence: float

@dataclass
class Plan:
    steps: List[PlanStep]
    root_node: MCTSNode

    def update(self, result: Dict):
        """Update plan based on execution result."""
        pass