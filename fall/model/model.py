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

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits