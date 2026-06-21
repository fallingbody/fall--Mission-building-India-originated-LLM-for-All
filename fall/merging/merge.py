"""
Automated model merging for FALL.
Supports weight interpolation, task vector arithmetic, and adapter fusion.
"""
import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MergeConfig:
    method: str = "linear"  # linear, task_vector, ties, dare
    coefficient: float = 0.5
    density: float = 1.0
    epsilon: float = 1e-8

class ModelMerger:
    def __init__(self, config: MergeConfig = None):
        self.config = config or MergeConfig()
    
    def linear_merge(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        alpha: float = 0.5,
    ) -> nn.Module:
        """Simple linear interpolation: merged = (1-alpha)*A + alpha*B."""
        merged = copy.deepcopy(model_a)
        merged_sd = merged.state_dict()
        b_sd = model_b.state_dict()
        
        for key in merged_sd:
            if key in b_sd:
                merged_sd[key] = (1 - alpha) * merged_sd[key] + alpha * b_sd[key]
        
        merged.load_state_dict(merged_sd)
        return merged
    
    def task_vector_merge(
        self,
        base_model: nn.Module,
        fine_tuned_model: nn.Module,
        coefficient: float = 1.0,
    ) -> nn.Module:
        """Task vector addition: merged = base + coeff * (finetuned - base)."""
        merged = copy.deepcopy(base_model)
        merged_sd = merged.state_dict()
        ft_sd = fine_tuned_model.state_dict()
        base_sd = base_model.state_dict()
        
        for key in merged_sd:
            if key in ft_sd and key in base_sd:
                task_vector = ft_sd[key] - base_sd[key]
                merged_sd[key] = base_sd[key] + coefficient * task_vector
        
        merged.load_state_dict(merged_sd)
        return merged
    
    def ties_merge(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        density: float = 0.2,
    ) -> nn.Module:
        """
        TIES (Trim, Elect Sign, Merge) merging.
        Handles interference between task vectors.
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Step 1: Compute task vectors and trim small values
        task_vectors = []
        base_sd = models[0].state_dict()
        for model in models[1:]:
            tv = {}
            sd = model.state_dict()
            for key in base_sd:
                if key in sd:
                    tv[key] = sd[key] - base_sd[key]
            task_vectors.append(self._trim(tv, density))
        
        # Step 2: Elect signs (resolve disagreements)
        elected = self._elect_signs(task_vectors)
        
        # Step 3: Merge elected vectors
        merged = copy.deepcopy(models[0])
        merged_sd = merged.state_dict()
        
        for key in merged_sd:
            if key in elected:
                merged_sd[key] = base_sd[key]
                for tv, w in zip(task_vectors, weights):
                    if key in tv:
                        mask = elected[key]
                        merged_sd[key] += w * tv[key] * mask
        
        merged.load_state_dict(merged_sd)
        return merged
    
    def _trim(self, task_vector: Dict[str, torch.Tensor], density: float) -> Dict[str, torch.Tensor]:
        """Keep only top density% of values by magnitude."""
        if density >= 1.0:
            return task_vector
        
        trimmed = {}
        for key, tensor in task_vector.items():
            flat = tensor.view(-1)
            k = max(1, int(len(flat) * density))
            threshold = torch.topk(flat.abs(), k, largest=True).values[-1]
            trimmed[key] = torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
        
        return trimmed
    
    def _elect_signs(self, task_vectors: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Elect sign based on total magnitude per direction."""
        elected = {}
        all_keys = set()
        for tv in task_vectors:
            all_keys.update(tv.keys())
        
        for key in all_keys:
            # Sum the magnitudes in positive and negative directions
            pos_sum = 0
            neg_sum = 0
            for tv in task_vectors:
                if key in tv:
                    pos_sum += tv[key].clamp(min=0).sum()
                    neg_sum += tv[key].clamp(max=0).abs().sum()
            
            # Elect positive if positive magnitude dominates
            if pos_sum >= neg_sum:
                elected[key] = torch.ones_like(task_vectors[0][key])
            else:
                elected[key] = -torch.ones_like(task_vectors[0][key])
        
        return elected
    
    def dare_merge(
        self,
        base_model: nn.Module,
        fine_tuned_model: nn.Module,
        drop_rate: float = 0.9,
        rescale: bool = True,
    ) -> nn.Module:
        """
        DARE (Drop And REscale) merging.
        Randomly drops delta parameters and rescales survivors.
        """
        merged = copy.deepcopy(base_model)
        merged_sd = merged.state_dict()
        ft_sd = fine_tuned_model.state_dict()
        base_sd = base_model.state_dict()
        
        for key in merged_sd:
            if key in ft_sd and key in base_sd:
                delta = ft_sd[key] - base_sd[key]
                
                # Random drop
                mask = torch.bernoulli(torch.ones_like(delta) * (1 - drop_rate))
                
                if rescale:
                    delta = delta * mask / max(1 - drop_rate, self.config.epsilon)
                else:
                    delta = delta * mask
                
                merged_sd[key] = base_sd[key] + delta
        
        merged.load_state_dict(merged_sd)
        return merged
    
    def fuse_adapters(
        self,
        base_model: nn.Module,
        adapters: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ) -> nn.Module:
        """Fuse multiple LoRA adapters into the base model."""
        if weights is None:
            weights = {k: 1.0 / len(adapters) for k in adapters}
        
        merged = copy.deepcopy(base_model)
        merged_sd = merged.state_dict()
        
        for adapter_name, adapter in adapters.items():
            adapter_sd = adapter.state_dict()
            weight = weights.get(adapter_name, 1.0)
            
            for key in merged_sd:
                if key in adapter_sd:
                    merged_sd[key] += weight * adapter_sd[key]
        
        merged.load_state_dict(merged_sd)
        return merged


class EnsemblePredictor:
    """Ensemble multiple models at inference time."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Weighted average of model outputs."""
        outputs = []
        for model, weight in zip(self.models, self.weights):
            logits = model(input_ids)
            outputs.append(F.softmax(logits, dim=-1) * weight)
        
        return torch.stack(outputs).sum(dim=0)