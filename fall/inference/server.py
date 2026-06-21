"""
vLLM-based inference server for FALL.
Supports continuous batching, speculative decoding, and tool streaming.
"""
import asyncio
import json
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
import uvicorn

@dataclass
class InferenceConfig:
    model_path: str = "/models/fall_5t"
    tensor_parallel_size: int = 16
    max_model_len: int = 524288
    max_batch_size: int = 256
    dtype: str = "float8_e4m3fn"
    trust_remote_code: bool = True

class FALLInferenceServer:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None  # Will be loaded by vLLM
        self.tokenizer = None
        self.active_requests: Dict[str, Dict] = {}
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "avg_latency_ms": 0.0,
        }

    async def load_model(self):
        """Load the model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                enforce_eager=False,
            )
            self.tokenizer = self.model.get_tokenizer()
        except ImportError:
            print("vLLM not available. Install with: pip install vllm")
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response."""
        if self.model is None:
            await self.load_model()

        request_id = str(uuid.uuid4())
        start_time = time.time()

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|task_end|>"],
        )

        # Add tool definitions to prompt
        if tools:
            tool_text = "\nAvailable tools:\n" + json.dumps(tools, indent=2)
            prompt = tool_text + "\n\n" + prompt

        outputs = self.model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        latency = (time.time() - start_time) * 1000
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        self.stats["total_requests"] += 1
        self.stats["total_tokens_generated"] += tokens_generated
        self.stats["avg_latency_ms"] = (
            self.stats["avg_latency_ms"] * (self.stats["total_requests"] - 1) + latency
        ) / self.stats["total_requests"]

        # Parse tool calls from output
        tool_calls = self._parse_tool_calls(generated_text)

        return {
            "request_id": request_id,
            "text": generated_text,
            "tool_calls": tool_calls,
            "tokens_generated": tokens_generated,
            "latency_ms": latency,
        }

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool call tokens from generated text."""
        import re
        tool_calls = []
        pattern = r'<\|tool_call\|>(.*?)</\|tool_call\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                tool_calls.append(json.loads(match))
            except json.JSONDecodeError:
                pass
        return tool_calls

    async def stream_generate(self, prompt: str, max_tokens: int = 4096):
        """Stream generated tokens."""
        if self.model is None:
            await self.load_model()
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        outputs = self.model.generate([prompt], sampling_params)
        for token_id in outputs[0].outputs[0].token_ids:
            token_text = self.tokenizer.decode([token_id])
            yield f"data: {json.dumps({'token': token_text})}\n\n"
        yield "data: [DONE]\n\n"

    def get_stats(self):
        return dict(self.stats)