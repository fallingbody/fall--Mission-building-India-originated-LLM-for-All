"""
Advanced decoding strategies for FALL.
Beam search, contrastive search, nucleus sampling, and more.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import math

@dataclass
class DecodingConfig:
    strategy: str = "nucleus"  # greedy, beam, nucleus, contrastive, speculative
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    num_beams: int = 4
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True

class BeamSearchDecoder:
    """Beam search with length and repetition penalties."""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
        self.num_beams = config.num_beams
    
    @torch.no_grad()
    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        tokenizer,
        eos_token_id: int = 2,
    ) -> List[Tuple[List[int], float]]:
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Expand for beam search
        input_ids = input_ids.repeat_interleave(self.num_beams, dim=0)
        beam_scores = torch.zeros(B * self.num_beams, device=device)
        beam_scores[1::self.num_beams] = -1e9  # Only keep first beam initially
        beam_scores = beam_scores.view(B, self.num_beams)
        
        for step in range(self.config.max_tokens):
            if input_ids.shape[1] > 524288:
                input_ids = input_ids[:, -524288:]
            
            logits = model(input_ids)[:, -1, :]
            logits = logits.view(B, self.num_beams, -1)
            
            # Apply temperature
            logits = logits / self.config.temperature
            
            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, input_ids.view(B, self.num_beams, -1)
                )
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add beam scores
            scores = beam_scores.unsqueeze(-1) + log_probs
            
            if step == 0:
                scores = scores[:, 0, :].unsqueeze(1)
            
            # Top-k selection
            topk_scores, topk_tokens = torch.topk(
                scores.view(B, -1),
                k=self.num_beams * 2,
                dim=-1,
            )
            
            # Select top beams
            beam_indices = topk_tokens // logits.shape[-1]
            token_indices = topk_tokens % logits.shape[-1]
            
            # Update
            beam_scores = topk_scores[:, :self.num_beams]
            token_indices = token_indices[:, :self.num_beams]
            beam_indices = beam_indices[:, :self.num_beams]
            
            # Reorder input_ids
            new_input_ids = []
            for b in range(B):
                batch_indices = beam_indices[b] + b * self.num_beams
                new_input_ids.append(input_ids[batch_indices])
            
            input_ids = torch.stack(new_input_ids).view(B * self.num_beams, -1)
            next_tokens = token_indices.view(-1, 1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            # Check for EOS
            if (token_indices == eos_token_id).any():
                break
        
        # Return best beams
        best_beams = []
        for b in range(B):
            best_idx = beam_scores[b].argmax().item()
            tokens = input_ids[b * self.num_beams + best_idx].tolist()
            score = beam_scores[b, best_idx].item()
            best_beams.append((tokens, score))
        
        return best_beams
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(prev_tokens.shape[2]):
            tokens = prev_tokens[:, :, i].unsqueeze(-1)
            penalty = torch.where(
                logits.gather(-1, tokens) > 0,
                self.config.repetition_penalty,
                1.0 / self.config.repetition_penalty,
            )
            logits.scatter_(-1, tokens, logits.gather(-1, tokens) / penalty)
        return logits

class NucleusDecoder:
    """Nucleus (top-p) sampling."""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
    
    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits / self.config.temperature
        
        if self.config.top_k > 0:
            topk_vals, _ = torch.topk(logits, self.config.top_k, dim=-1)
            logits[logits < topk_vals[:, -1:]] = float('-inf')
        
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative > self.config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

class ContrastiveDecoder:
    """Contrastive decoding — penalizes expert model with amateur model outputs."""
    
    def __init__(
        self,
        expert_model,
        amateur_model,
        alpha: float = 0.1,
    ):
        self.expert = expert_model
        self.amateur = amateur_model
        self.alpha = alpha
    
    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor) -> torch.Tensor:
        expert_logits = self.expert(input_ids)[:, -1, :]
        amateur_logits = self.amateur(input_ids)[:, -1, :]
        
        # Contrastive penalty
        contrastive_logits = expert_logits - self.alpha * amateur_logits
        
        probs = F.softmax(contrastive_logits, dim=-1)
        return torch.multinomial(probs, 1)