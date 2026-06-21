"""
A/B Testing framework for FALL model versions.
Compares multiple model versions in production with traffic splitting.
"""
import random
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ABTest:
    name: str
    variants: Dict[str, str]  # variant_name -> model_version_id
    traffic_split: Dict[str, float]  # variant_name -> fraction of traffic
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

class ABTestingEngine:
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: List[ABTest] = []
    
    def create_test(
        self,
        name: str,
        variants: Dict[str, str],
        traffic_split: Optional[Dict[str, float]] = None,
    ) -> ABTest:
        if traffic_split is None:
            equal_split = 1.0 / len(variants)
            traffic_split = {v: equal_split for v in variants}
        
        test = ABTest(
            name=name,
            variants=variants,
            traffic_split=traffic_split,
        )
        self.active_tests[name] = test
        return test
    
    def route_request(self, test_name: str) -> str:
        """Route a request to a variant based on traffic split."""
        test = self.active_tests.get(test_name)
        if not test:
            return "default"
        
        r = random.random()
        cumulative = 0.0
        for variant, fraction in test.traffic_split.items():
            cumulative += fraction
            if r <= cumulative:
                return variant
        return list(test.variants.keys())[-1]
    
    def record_metric(self, test_name: str, variant: str, metric_name: str, value: float):
        """Record a metric for a specific variant."""
        test = self.active_tests.get(test_name)
        if not test:
            return
        
        if variant not in test.metrics:
            test.metrics[variant] = {}
        if metric_name not in test.metrics[variant]:
            test.metrics[variant][metric_name] = []
        test.metrics[variant][metric_name].append(value)
    
    def evaluate_test(self, test_name: str) -> Dict[str, Any]:
        """Evaluate A/B test results."""
        test = self.active_tests.get(test_name)
        if not test:
            return {}
        
        results = {}
        for variant, metrics in test.metrics.items():
            results[variant] = {
                metric: {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
                for metric, values in metrics.items()
            }
        return results
    
    def complete_test(self, test_name: str, winning_variant: str):
        """End a test and declare a winner."""
        test = self.active_tests.pop(test_name, None)
        if test:
            self.completed_tests.append(test)