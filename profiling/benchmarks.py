"""
Micro-benchmarks for FALL components.
"""
import torch
import time
from fall.profiling.profiler import profiler
from fall.model.config import FALLConfig
from fall.model.attention import (
    MultiHeadLatentAttention,
    SSDBlock,
    HyperbolicAttention,
    FourierNeuralOperator,
)
from fall.model.moe import AuxiliaryLossFreeMoE

class MicroBenchmarks:
    def __init__(self, config: FALLConfig = None):
        self.config = config or FALLConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def benchmark_attention(self, batch_size=2, seq_len=2048, warmup=10, iterations=100):
        """Benchmark MLA attention."""
        attn = MultiHeadLatentAttention(self.config).to(self.device)
        x = torch.randn(batch_size, seq_len, self.config.d_model, device=self.device)
        
        # Warmup
        for _ in range(warmup):
            attn(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            attn(x)
        torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / iterations * 1000
        print(f"MLA Attention ({seq_len} tokens): {elapsed:.2f}ms per forward")
        return elapsed
    
    def benchmark_moe(self, batch_size=2, seq_len=2048, warmup=10, iterations=100):
        """Benchmark MoE layer."""
        moe = AuxiliaryLossFreeMoE(self.config).to(self.device)
        x = torch.randn(batch_size, seq_len, self.config.d_model, device=self.device)
        
        for _ in range(warmup):
            moe(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            moe(x)
        torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / iterations * 1000
        print(f"MoE ({self.config.n_experts_per_layer} experts): {elapsed:.2f}ms per forward")
        return elapsed
    
    def benchmark_ssd(self, batch_size=2, seq_len=8192, warmup=10, iterations=100):
        """Benchmark SSD/Mamba block (linear scaling)."""
        ssd = SSDBlock(self.config).to(self.device)
        x = torch.randn(batch_size, seq_len, self.config.d_model, device=self.device)
        
        for _ in range(warmup):
            ssd(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            ssd(x)
        torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / iterations * 1000
        tokens_per_sec = (batch_size * seq_len * iterations) / ((time.time() - start) if time.time() > start else 1)
        print(f"SSD Block ({seq_len} tokens): {elapsed:.2f}ms, {tokens_per_sec:,.0f} tokens/s")
        return elapsed
    
    def run_all(self):
        print("Running FALL Micro-Benchmarks...\n")
        results = {}
        results["attention"] = self.benchmark_attention()
        results["moe"] = self.benchmark_moe()
        results["ssd"] = self.benchmark_ssd()
        return results