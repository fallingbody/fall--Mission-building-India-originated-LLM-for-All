"""
FALL Python SDK.
Three lines to command 5 trillion parameters.
"""
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import time

@dataclass
class FALLResponse:
    task_id: str
    status: str
    result: Dict[str, Any]
    duration_seconds: float
    tokens_generated: int

class FALLClient:
    """
    FALL API Client.
    
    Usage:
        client = FALLClient(api_key="sk-your-key")
        result = client.execute("hack 192.168.1.50")
        print(result.result)
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.fall.ai",
        timeout: int = 3600,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        })
    
    def execute(
        self,
        task: str,
        mode: str = "autonomous",
        tools: Optional[List[str]] = None,
        max_duration: int = 3600,
    ) -> FALLResponse:
        """Execute an autonomous task."""
        payload = {
            "task": task,
            "mode": mode,
            "max_duration": max_duration,
        }
        if tools:
            payload["tools"] = tools
        
        start = time.time()
        response = self.session.post(
            f"{self.base_url}/v1/execute",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        return FALLResponse(
            task_id=data["task_id"],
            status=data["status"],
            result=data["result"],
            duration_seconds=time.time() - start,
            tokens_generated=data.get("tokens_generated", 0),
        )
    
    def hack(self, target: str, mode: str = "autonomous") -> FALLResponse:
        """Execute a penetration test against a target."""
        return self.execute(
            task=f"Penetration test {target}. Find vulnerabilities, exploit them, and report.",
            mode=mode,
            tools=["nmap", "metasploit", "hashcat", "hydra", "shell", "python"],
        )
    
    def defend(self, target: str = "localhost") -> FALLResponse:
        """Harden and defend a system."""
        return self.execute(
            task=f"Audit and harden {target}. Find vulnerabilities, patch them, configure defenses.",
            mode="autonomous",
            tools=["shell", "python", "nmap", "file_read", "file_write"],
        )
    
    def analyze_code(self, code: str, language: str = "auto") -> FALLResponse:
        """Analyze code for vulnerabilities."""
        return self.execute(
            task=f"Analyze this code for vulnerabilities:\n```{language}\n{code}\n```\nFind all security issues and suggest fixes.",
            mode="autonomous",
        )
    
    def write_code(self, specification: str, language: str = "python") -> FALLResponse:
        """Write code from specification."""
        return self.execute(
            task=f"Write {language} code that: {specification}. Include tests and documentation.",
            mode="autonomous",
        )
    
    def ask(self, question: str) -> FALLResponse:
        """Ask a general question."""
        return self.execute(task=question, mode="interactive")
    
    def stream_execute(self, task: str, mode: str = "autonomous"):
        """Stream execution results."""
        payload = {"task": task, "mode": mode}
        response = self.session.post(
            f"{self.base_url}/v1/execute/stream",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/v1/stats")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if FALL is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class FALLAsyncClient:
    """Async version of FALLClient."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.fall.ai"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession(headers={"X-API-Key": self.api_key})
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def execute(self, task: str, mode: str = "autonomous") -> FALLResponse:
        payload = {"task": task, "mode": mode}
        async with self.session.post(
            f"{self.base_url}/v1/execute",
            json=payload,
        ) as resp:
            data = await resp.json()
            return FALLResponse(**data)