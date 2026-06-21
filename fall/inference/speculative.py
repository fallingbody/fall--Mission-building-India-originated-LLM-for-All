"""
Speculative decoding for FALL inference.
Uses a small draft model to accelerate generation 3-5x.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class SpeculativeConfig:
    draft_model_path: str = "/models/fall_draft_100m"
    draft_vocab_size: int = 256_000
    draft_d_model: int = 1024
    draft_n_layers: int = 8
    draft_n_heads: int = 16
    num_speculative_tokens: int = 5
    acceptance_threshold: float = 0.9

class FALLDraftModel(nn.Module):
    """Lightweight draft model (~100M params) for speculative decoding."""
    def __init__(self, config: SpeculativeConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.draft_vocab_size, config.draft_d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.draft_d_model,
                nhead=config.draft_n_heads,
                dim_feedforward=config.draft_d_model * 4,
                batch_first=True,
            )
            for _ in range(config.draft_n_layers)
        ])
        self.lm_head = nn.Linear(config.draft_d_model, config.draft_vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def generate_draft(self, input_ids: torch.Tensor, k: int) -> Tuple[List[int], torch.Tensor]:
        """Generate k draft tokens autoregressively."""
        draft_tokens = []
        current = input_ids.clone()
        logprobs = []

        for _ in range(k):
            with torch.no_grad():
                logits = self.forward(current)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                draft_tokens.append(next_token)
                logprobs.append(torch.log(probs[next_token]))
                current = torch.cat([current, torch.tensor([[next_token]])], dim=1)

        return draft_tokens, torch.stack(logprobs)


class SpeculativeDecoder:
    """Manages speculative decoding for FALL."""
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: FALLDraftModel,
        tokenizer,
        config: SpeculativeConfig,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "total_drafts": 0,
            "total_verifications": 0,
        }

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Tuple[str, dict]:
        """Generate text with speculative decoding."""
        start_time = time.time()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        generated_ids = []

        while len(generated_ids) < max_new_tokens:
            # Draft k tokens with small model
            draft_tokens, draft_logprobs = self.draft_model.generate_draft(
                input_ids, self.config.num_speculative_tokens
            )
            self.stats["total_drafts"] += 1

            if not draft_tokens:
                break

            # Verify all k tokens in one target forward pass
            full_input = torch.cat([
                input_ids,
                torch.tensor([draft_tokens])
            ], dim=1)

            with torch.no_grad():
                target_logits = self.target_model(full_input)
                target_probs = F.softmax(
                    target_logits[:, -self.config.num_speculative_tokens - 1:, :],
                    dim=-1
                )

            # Accept/reject each draft token
            accepted = 0
            for i, draft_token in enumerate(draft_tokens):
                target_prob = target_probs[0, i, draft_token].item()
                draft_prob = torch.exp(draft_logprobs[i]).item()

                # Acceptance probability
                accept_prob = min(1.0, target_prob / max(draft_prob, 1e-8))

                if accept_prob > self.config.acceptance_threshold or target_prob > draft_prob:
                    generated_ids.append(draft_token)
                    accepted += 1
                else:
                    # Rejection sampling: sample from residual distribution
                    residual = target_probs[0, i] - F.softmax(
                        torch.exp(draft_logprobs[i]), dim=-1
                    ).clamp(min=0)
                    residual = residual / residual.sum()
                    new_token = torch.multinomial(residual, 1).item()
                    generated_ids.append(new_token)
                    break

            self.stats["total_tokens"] += self.config.num_speculative_tokens
            self.stats["accepted_tokens"] += accepted
            self.stats["total_verifications"] += 1

            # Update input for next iteration
            input_ids = torch.cat([
                input_ids,
                torch.tensor([generated_ids[-accepted:]])
            ], dim=1) if accepted > 0 else input_ids

        # Decode
        text = self.tokenizer.decode(generated_ids)

        stats = {
            "acceptance_rate": self.stats["accepted_tokens"] / max(1, self.stats["total_tokens"]),
            "speedup": self.stats["accepted_tokens"] / max(1, self.stats["total_verifications"]),
            "latency_seconds": time.time() - start_time,
            "tokens_per_second": len(generated_ids) / max(0.001, time.time() - start_time),
        }

        return text, stats