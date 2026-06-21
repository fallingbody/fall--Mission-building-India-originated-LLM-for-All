"""
Intelligent resource scheduler for FALL.
Manages GPU allocation, task prioritization, and energy efficiency.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq

@dataclass
class GPUNode:
    id: str
    total_memory_gb: float
    available_memory_gb: float
    utilization: float
    temperature: float
    power_watts: float
    status: str  # active, idle, maintenance, offline

@dataclass
class ComputeTask:
    priority: int
    gpu_requirement: float
    memory_requirement: float
    estimated_duration: float
    task_fn: callable
    task_args: tuple

class ResourceScheduler:
    def __init__(self, gpu_nodes: List[GPUNode]):
        self.nodes = {n.id: n for n in gpu_nodes}
        self.task_queue: List[Tuple[int, int, ComputeTask]] = []
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.task_counter = 0
        self.energy_budget_watts = float('inf')
        self.current_power = 0.0
    
    def schedule(self, task: ComputeTask) -> Optional[str]:
        """Schedule a task on the best GPU."""
        best_node = self._find_best_node(task)
        if best_node is None:
            self.task_counter += 1
            heapq.heappush(
                self.task_queue,
                (-task.priority, self.task_counter, task)
            )
            return None
        
        self._allocate(best_node, task)
        return best_node.id
    
    def _find_best_node(self, task: ComputeTask) -> Optional[GPUNode]:
        candidates = []
        for node in self.nodes.values():
            if node.status != "active":
                continue
            if node.available_memory_gb < task.memory_requirement:
                continue
            if self.current_power + node.power_watts > self.energy_budget_watts:
                continue
            
            score = node.available_memory_gb - task.memory_requirement
            score -= node.utilization * 0.1
            score -= node.temperature * 0.01
            candidates.append((score, node))
        
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]
    
    def _allocate(self, node: GPUNode, task: ComputeTask):
        node.available_memory_gb -= task.memory_requirement
        node.utilization += task.gpu_requirement
        self.current_power += node.power_watts
        self.active_tasks[f"{node.id}_{self.task_counter}"] = task
    
    def release(self, task_id: str, node_id: str):
        if node_id in self.nodes:
            node = self.nodes[node_id]
            task = self.active_tasks.pop(task_id, None)
            if task:
                node.available_memory_gb += task.memory_requirement
                node.utilization -= task.gpu_requirement
                self.current_power -= node.power_watts
        self._process_queue()
    
    def _process_queue(self):
        while self.task_queue:
            _, _, task = heapq.heappop(self.task_queue)
            node = self._find_best_node(task)
            if node:
                self._allocate(node, task)
            else:
                heapq.heappush(
                    self.task_queue,
                    (-task.priority, self.task_counter, task)
                )
                break
    
    def set_energy_budget(self, watts: float):
        self.energy_budget_watts = watts
    
    def get_cluster_status(self) -> Dict:
        return {
            "total_nodes": len(self.nodes),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "total_power_watts": self.current_power,
            "nodes": [
                {"id": n.id, "util": n.utilization, "mem_free": n.available_memory_gb}
                for n in self.nodes.values()
            ],
        }