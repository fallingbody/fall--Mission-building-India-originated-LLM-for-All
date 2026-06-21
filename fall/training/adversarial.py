"""
Adversarial training for FALL.
Trains model to be robust against input perturbations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialTrainer:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        iterations: int = 3,
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
    
    def fgsm_attack(self, input_ids: torch.Tensor, loss_fn, target_ids: torch.Tensor):
        """Fast Gradient Sign Method attack."""
        embedding = self.model.embed(input_ids)
        embedding.requires_grad = True
        
        logits = self.model.layers(embedding)
        loss = loss_fn(logits, target_ids)
        loss.backward()
        
        perturbed = embedding + self.epsilon * embedding.grad.sign()
        return perturbed.detach()
    
    def pgd_attack(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Projected Gradient Descent attack on embeddings."""
        original = self.model.embed(input_ids).detach()
        perturbed = original.clone().detach() + torch.randn_like(original) * 0.001
        
        for _ in range(self.iterations):
            perturbed.requires_grad = True
            
            # Forward pass (simplified)
            logits = self.model(perturbed)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            
            # Step
            with torch.no_grad():
                perturbed = perturbed + self.alpha * perturbed.grad.sign()
                # Project
                delta = torch.clamp(perturbed - original, -self.epsilon, self.epsilon)
                perturbed = original + delta
        
        return perturbed.detach()
    
    def train_adversarial_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One adversarial training step."""
        # Normal step
        logits = self.model(input_ids)
        clean_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        
        # Adversarial step
        adv_embeddings = self.fgsm_attack(input_ids, F.cross_entropy, labels)
        adv_logits = self.model(adv_embeddings)
        adv_loss = F.cross_entropy(
            adv_logits.view(-1, adv_logits.size(-1)),
            labels.view(-1),
        )
        
        # Combined loss
        loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()