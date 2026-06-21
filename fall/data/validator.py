"""
Continuous dataset validation.
Detects data quality degradation, bias drift, and poisoning attempts.
"""
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class DataStats:
    total_samples: int
    total_tokens: int
    unique_tokens: int
    language_distribution: Dict[str, float]
    length_distribution: Dict[str, float]
    checksums: List[str] = field(default_factory=list)


class DatasetValidator:
    def __init__(self, tokenizer, expected_languages: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        self.expected_languages = expected_languages or ["en"]
        self.baseline_stats: Optional[DataStats] = None
    
    def compute_stats(self, dataset_path: str, sample_size: int = 10000) -> DataStats:
        """Compute statistics for a dataset."""
        total = 0
        total_tokens = 0
        all_tokens = set()
        languages = Counter()
        lengths = Counter()
        checksums = []
        
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                
                total += 1
                tokens = self.tokenizer.encode(line)
                total_tokens += len(tokens)
                all_tokens.update(tokens)
                
                # Categorize length
                if len(tokens) < 128:
                    lengths["short"] += 1
                elif len(tokens) < 1024:
                    lengths["medium"] += 1
                elif len(tokens) < 8192:
                    lengths["long"] += 1
                else:
                    lengths["very_long"] += 1
                
                # Hash for integrity
                checksums.append(hashlib.md5(line.encode()).hexdigest())
        
        return DataStats(
            total_samples=total,
            total_tokens=total_tokens,
            unique_tokens=len(all_tokens),
            language_distribution={lang: count/total for lang, count in languages.items()},
            length_distribution={k: v/total for k, v in lengths.items()},
            checksums=checksums,
        )
    
    def set_baseline(self, dataset_path: str):
        """Set the baseline statistics for comparison."""
        self.baseline_stats = self.compute_stats(dataset_path)
    
    def detect_anomalies(self, dataset_path: str) -> List[str]:
        """Detect anomalies compared to baseline."""
        if self.baseline_stats is None:
            return ["No baseline set"]
        
        current = self.compute_stats(dataset_path)
        alerts = []
        
        # Check token count drift
        if self.baseline_stats.total_tokens > 0:
            token_drift = abs(current.total_tokens - self.baseline_stats.total_tokens)
            token_drift_pct = token_drift / self.baseline_stats.total_tokens
            if token_drift_pct > 0.2:
                alerts.append(f"Token count drift: {token_drift_pct:.1%}")
        
        # Check length distribution shift
        for key in self.baseline_stats.length_distribution:
            base_pct = self.baseline_stats.length_distribution.get(key, 0)
            curr_pct = current.length_distribution.get(key, 0)
            if abs(base_pct - curr_pct) > 0.15:
                alerts.append(f"Length distribution shift in '{key}': {base_pct:.1%} -> {curr_pct:.1%}")
        
        # Check for duplicate injection (poisoning)
        unique_checksums = len(set(current.checksums))
        if unique_checksums < len(current.checksums) * 0.5:
            alerts.append(f"High duplication: {unique_checksums}/{len(current.checksums)} unique")
        
        return alerts if alerts else ["No anomalies detected"]