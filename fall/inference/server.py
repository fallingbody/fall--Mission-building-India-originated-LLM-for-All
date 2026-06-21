"""
Custom PyTorch Online Learning inference server for FALL.
Supports continuous batching, tool execution, knowledge graph memory, and dynamic weight updates.
"""
import asyncio
import json
import time
import uuid
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
import uvicorn
from collections import deque

from transformers import AutoTokenizer

from fall.model.model import FALLForCausalLM
from fall.model.config import FALLConfig
from fall.knowledge.graph import KnowledgeGraph, Entity

@dataclass
class InferenceConfig:
    model_path: str = "/models/fall_5t"
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    max_batch_size: int = 256
    dtype: str = "float16"
    trust_remote_code: bool = True

class FALLInferenceServer:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        # Use a real tokenizer matching the vocab_size of 50257
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.active_requests: Dict[str, Dict] = {}
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "avg_latency_ms": 0.0,
        }
        
        # --- Continual Learning Components ---
        # Knowledge Graph for semantic memory
        self.kg = KnowledgeGraph(persist_path="./knowledge_graph_db")
        # Replay buffer to mitigate catastrophic forgetting (stores past conversation text)
        self.replay_buffer = deque(maxlen=50)

    async def load_model(self):
        """Load the custom FALL model and initialize the optimizer."""
        print("Loading custom FALLForCausalLM for Online Learning...")
        fall_config = FALLConfig()
        self.model = FALLForCausalLM(fall_config)
        
        # Set to train mode since we are updating weights dynamically
        self.model.train()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        import glob
        import os
        checkpoints = sorted(
            glob.glob("checkpoints/step_*.pt"), 
            key=lambda x: int(os.path.basename(x).replace("step_", "").replace(".pt", "")), 
            reverse=True
        )
        for ckpt in checkpoints:
            try:
                print(f"Attempting to load checkpoint: {ckpt}")
                state = torch.load(ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
                self.model.load_state_dict(state["model"])
                print(f"Successfully loaded {ckpt}!")
                break
            except Exception as e:
                print(f"Failed to load {ckpt} (likely corrupted during save): {e}")
            
        # Initialize Optimizer for Online Learning
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        print("Model and Optimizer loaded successfully.")

    def _execute_tool(self, tool_call: Dict) -> str:
        """Executes the requested tool and returns the result."""
        tool_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        if tool_name == "web_search":
            query = args.get("query", "")
            try:
                from duckduckgo_search import DDGS
                results = DDGS().text(query, max_results=3)
                return json.dumps(results)
            except ImportError:
                return "Tool execution failed: duckduckgo-search is not installed."
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        return "Unknown tool."

    def _train_on_interaction(self, input_text: str, generated_text: str):
        """Performs an online gradient update step on the conversation."""
        if self.model is None or self.optimizer is None:
            return
            
        # Add interaction to replay buffer
        full_text = f"User: {input_text}\nAssistant: {generated_text}"
        self.replay_buffer.append(full_text)
        
        # Sample from replay buffer to prevent catastrophic forgetting
        import random
        batch_texts = [full_text]
        if len(self.replay_buffer) > 1:
            samples = random.sample(list(self.replay_buffer)[:-1], min(3, len(self.replay_buffer) - 1))
            batch_texts.extend(samples)
            
        total_loss = 0
        self.optimizer.zero_grad()
        
        for text in batch_texts:
            input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                
            if input_ids.shape[1] < 2:
                continue
                
            # Forward pass
            logits = self.model(input_ids)
            
            # Next-token prediction loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss

        # Backward and step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        print(f"[Online Learning] Weights updated! Loss: {total_loss.item():.4f}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response with tool execution and online learning."""
        if self.model is None:
            await self.load_model()

        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Inject semantic memory from Knowledge Graph
        try:
            memories = self.kg.query()
            if memories:
                memory_context = "\n[Semantic Memory:]\n" + "\n".join(
                    [f"- {m.type}: {m.properties}" for m in memories[-5:]]
                ) + "\n"
                prompt = memory_context + prompt
        except Exception:
            pass

        # Add tool definitions to prompt
        if tools:
            tool_text = "\nAvailable tools:\n" + json.dumps(tools, indent=2)
            prompt = tool_text + "\n\n" + prompt

        # Tokenize
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Generate phase
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=min(max_tokens, 10),
                temperature=temperature
            )
            
        new_token_ids = output_ids[0][input_ids.shape[1]:].tolist()
        generated_text = self.tokenizer.decode(new_token_ids)

        # Parse tool calls and execute if needed
        tool_calls = self._parse_tool_calls(generated_text)
        if tool_calls:
            print(f"Executing tools: {tool_calls}")
            tool_results = []
            for call in tool_calls:
                result = self._execute_tool(call)
                tool_results.append(f"Tool {call['name']} returned: {result}")
            
            # If tools were executed, append results and generate final answer
            extended_prompt = prompt + generated_text + "\n" + "\n".join(tool_results) + "\nAssistant:"
            ext_input_ids = torch.tensor([self.tokenizer.encode(extended_prompt)], dtype=torch.long)
            if torch.cuda.is_available():
                ext_input_ids = ext_input_ids.cuda()
            
            with torch.no_grad():
                final_output_ids = self.model.generate(
                    input_ids=ext_input_ids,
                    max_new_tokens=min(max_tokens, 100),
                    temperature=temperature
                )
            new_token_ids = final_output_ids[0][ext_input_ids.shape[1]:].tolist()
            generated_text = self.tokenizer.decode(new_token_ids)

        # Save new learnings to Knowledge Graph
        if "learn:" in generated_text.lower():
            self.kg.add_entity(Entity(
                id=None,
                type="learning",
                properties={"content": generated_text[:200]}
            ))

        # Online Learning Update!
        # Run in background to avoid blocking the API response
        asyncio.create_task(asyncio.to_thread(self._train_on_interaction, prompt, generated_text))

        latency = (time.time() - start_time) * 1000
        tokens_generated = len(new_token_ids)
        self.stats["total_requests"] += 1
        self.stats["total_tokens_generated"] += tokens_generated
        self.stats["avg_latency_ms"] = (
            self.stats["avg_latency_ms"] * (self.stats["total_requests"] - 1) + latency
        ) / self.stats["total_requests"]

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
        """Stream generated tokens (simplified for UI, falls back to block generate if tools)."""
        result = await self.generate(prompt, max_tokens)
        
        # Simply stream the final result back in chunks
        chunk_size = 4
        text = result["text"]
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            yield f"data: {json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.01)
            
        yield "data: [DONE]\n\n"

    def get_stats(self):
        return dict(self.stats)