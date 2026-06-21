"""
Reinforcement Learning from Human Feedback (RLHF) and DPO for FALL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RLHFConfig:
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    ppo_epsilon: float = 0.2
    ppo_clip_value: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"  # sigmoid, hinge, ipo

class PPOTrainer:
    """Proximal Policy Optimization for FALL."""
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        config: RLHFConfig = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.config = config or RLHFConfig()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def train_step(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """One PPO training step."""
        # Get log probs from current and reference models
        with torch.no_grad():
            ref_logits = self.reference_model(queries)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_response_log_probs = torch.gather(
                ref_log_probs, -1, responses.unsqueeze(-1)
            ).squeeze(-1)
        
        logits = self.model(queries)
        log_probs = F.log_softmax(logits, dim=-1)
        response_log_probs = torch.gather(
            log_probs, -1, responses.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute ratio
        ratio = torch.exp(response_log_probs - ref_response_log_probs)
        
        # Compute advantages (normalize)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.ppo_clip_value,
            1 + self.config.ppo_clip_value,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(logits.max(dim=-1).values, rewards)
        
        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

class DPOTrainer:
    """Direct Preference Optimization for FALL."""
    
    def __init__(self, model: nn.Module, config: RLHFConfig = None):
        self.model = model
        self.config = config or RLHFConfig()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def train_step(
        self,
        chosen_inputs: torch.Tensor,
        rejected_inputs: torch.Tensor,
    ) -> Dict[str, float]:
        """One DPO training step."""
        # Get logits for chosen and rejected
        chosen_logits = self.model(chosen_inputs)
        rejected_logits = self.model(rejected_inputs)
        
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        
        # Compute DPO loss
        if self.config.dpo_loss_type == "sigmoid":
            # Standard sigmoid DPO loss
            chosen_logp = chosen_log_probs.mean()
            rejected_logp = rejected_log_probs.mean()
            
            log_ratio = chosen_logp - rejected_logp
            loss = -F.logsigmoid(self.config.dpo_beta * log_ratio).mean()
        
        elif self.config.dpo_loss_type == "hinge":
            chosen_logp = chosen_log_probs.mean()
            rejected_logp = rejected_log_probs.mean()
            
            log_ratio = chosen_logp - rejected_logp
            loss = F.relu(1 - self.config.dpo_beta * log_ratio).mean()
        
        elif self.config.dpo_loss_type == "ipo":
            chosen_logp = chosen_log_probs.mean()
            rejected_logp = rejected_log_probs.mean()
            
            log_ratio = chosen_logp - rejected_logp
            loss = (log_ratio - 1 / (2 * self.config.dpo_beta)) ** 2
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "dpo_loss": loss.item(),
            "chosen_logp": chosen_logp.item(),
            "rejected_logp": rejected_logp.item(),
        }