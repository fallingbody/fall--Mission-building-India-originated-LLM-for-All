"""
FastAPI server for FALL inference.
Accepts the three-line API call and returns results.
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import logging
import uvicorn

from fall.inference.server import FALLInferenceServer, InferenceConfig

app = FastAPI(title="FALL API", version="1.0.0")
api_key_header = APIKeyHeader(name="X-API-Key")

# API Keys
API_KEYS = {
    "sk-defender-abc": {"role": "defender", "rate_limit": 100, "name": "Blue Team Alpha"},
    "sk-pentest-xyz": {"role": "pentester", "rate_limit": 50, "name": "Red Team Beta"},
    "sk-admin-full": {"role": "admin", "rate_limit": 1000, "name": "System Administrator"},
    "sk-local-dev": {"role": "admin", "rate_limit": 1000, "name": "Local Development"},
}

# Tool schemas
TOOL_SCHEMAS = {
    "shell": {"description": "Execute shell command", "parameters": {"cmd": "str", "timeout": "int"}},
    "python": {"description": "Execute Python code", "parameters": {"code": "str"}},
    "nmap": {"description": "Network scan", "parameters": {"target": "str", "ports": "str"}},
    "metasploit": {"description": "Metasploit module", "parameters": {"module": "str", "target": "str"}},
    "file_read": {"description": "Read file", "parameters": {"path": "str"}},
    "file_write": {"description": "Write file", "parameters": {"path": "str", "content": "str"}},
}

class ExecuteRequest(BaseModel):
    task: str = Field(..., description="The task to execute")
    mode: str = Field(default="autonomous", description="autonomous | interactive")
    max_duration: Optional[int] = Field(default=3600, description="Max seconds")
    tools: Optional[List[str]] = Field(default=None, description="Tools to enable")

class ExecuteResponse(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any]
    duration_seconds: float
    tokens_generated: int

# Initialize inference server
inference_server = FALLInferenceServer(InferenceConfig())

@app.on_event("startup")
async def startup():
    await inference_server.load_model()

@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute_task(
    request: ExecuteRequest,
    api_key: str = Depends(api_key_header),
):
    """Execute an autonomous task."""
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    role = API_KEYS[api_key]["role"]
    if role == "pentester" and "metasploit" not in (request.tools or []):
        if request.tools is None:
            request.tools = []
        request.tools.append("metasploit")

    task_id = f"task_{int(time.time())}_{hash(request.task) % 10000:04d}"
    start_time = time.time()

    # Build prompt with tool definitions
    available_tools = request.tools or list(TOOL_SCHEMAS.keys())
    prompt = f"""<|system|>
You are FALL, an autonomous cybersecurity agent. Role: {role}.
Use <|think|> tags for internal reasoning.
Call tools using <|tool_call|> JSON blocks.
End tasks with <|task_end|>.
</|system|>

<|user|>
{request.task}
</|user|>

<|assistant|>
<|think|>I need to analyze this task and plan my approach.</|think|>
"""

    # Generate
    result = await inference_server.generate(
        prompt=prompt,
        max_tokens=8192,
        tools={k: TOOL_SCHEMAS[k] for k in available_tools if k in TOOL_SCHEMAS},
    )

    duration = time.time() - start_time

    return ExecuteResponse(
        task_id=task_id,
        status="complete",
        result={
            "output": result["text"],
            "tool_calls": result["tool_calls"],
            "tokens_generated": result["tokens_generated"],
        },
        duration_seconds=duration,
        tokens_generated=result["tokens_generated"],
    )

@app.get("/v1/stats")
async def get_stats(api_key: str = Depends(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403)
    return inference_server.get_stats()

@app.get("/health")
async def health():
    return {"status": "healthy"}

def start_server(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)