import torch
import torch.nn as nn
from .config import FALLConfig
from .layers import FALLDecoderLayer

class FALLForCausalLM(nn.Module):
    def __init__(self, config: FALLConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            FALLDecoderLayer(config, i+1) for i in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, attention_mask=None, is_reasoning_mode=False):
        B, L = input_ids.shape
        x = self.embed(input_ids)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones((L, L), device=x.device)).view(1, 1, L, L)
            
        for layer in self.layers:
            x = layer(x, attention_mask, is_reasoning_mode=is_reasoning_mode)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=50, repetition_penalty=1.2):
        """Generate sequence autoregressively."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            logits = self(idx_cond)
            # Only care about the last token
            logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(logits.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= repetition_penalty
                        else:
                            logits[i, token_id] /= repetition_penalty
            
            if temperature > 0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
        self.train()
        return input_ids