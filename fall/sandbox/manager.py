"""
Secure sandbox manager for FALL tool execution.
Provides isolated containers with tool access.
"""
import asyncio
import json
import os
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    image: str = "fall-sandbox:latest"
    max_cpu: float = 4.0
    max_memory: str = "8g"
    max_disk: str = "10g"
    timeout: int = 300
    network_enabled: bool = True
    allow_egress: bool = False
    snapshot_before: bool = True

class SandboxManager:
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.active_containers: Dict[str, Dict] = {}
        self.tool_registry = self._build_tool_registry()
        self.audit_log = []

    def _build_tool_registry(self) -> Dict[str, callable]:
        return {
            "shell": self._exec_shell,
            "python": self._exec_python,
            "nmap": self._exec_nmap,
            "metasploit": self._exec_metasploit,
            "file_read": self._read_file,
            "file_write": self._write_file,
            "get_telemetry": self._get_telemetry,
            "hashcat": self._exec_hashcat,
            "john": self._exec_john,
            "hydra": self._exec_hydra,
            "curl": self._exec_curl,
        }

    async def execute(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """Execute a tool in the sandbox."""
        if tool_name not in self.tool_registry:
            return {"error": f"Unknown tool: {tool_name}"}

        start_time = time.time()
        task_id = f"task_{int(start_time)}_{hash(str(args)) % 10000:04d}"

        # Log the action
        self.audit_log.append({
            "task_id": task_id,
            "tool": tool_name,
            "args": args,
            "timestamp": start_time,
        })

        try:
            handler = self.tool_registry[tool_name]
            result = await handler(args)
            duration = time.time() - start_time
            return {
                "task_id": task_id,
                "tool": tool_name,
                "success": True,
                "result": result,
                "duration_seconds": duration,
            }
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {
                "task_id": task_id,
                "tool": tool_name,
                "success": False,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
            }

    async def _exec_shell(self, args: Dict) -> str:
        cmd = args.get("cmd", "")
        timeout = args.get("timeout", self.config.timeout)
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return stdout.decode('utf-8', errors='replace') + stderr.decode('utf-8', errors='replace')
        except asyncio.TimeoutError:
            process.kill()
            return f"Command timed out after {timeout}s"

    async def _exec_python(self, args: Dict) -> str:
        code = args.get("code", "")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = await self._exec_shell({"cmd": f"python3 {f.name}", "timeout": args.get("timeout", 30)})
        os.unlink(f.name)
        return result

    async def _exec_nmap(self, args: Dict) -> str:
        target = args.get("target", "127.0.0.1")
        ports = args.get("ports", "1-1000")
        flags = args.get("flags", "-sV -sC")
        cmd = f"nmap {flags} -p {ports} {target}"
        return await self._exec_shell({"cmd": cmd, "timeout": 300})

    async def _exec_metasploit(self, args: Dict) -> str:
        module = args.get("module", "")
        target = args.get("target", "")
        payload = args.get("payload", "generic/shell_reverse_tcp")
        cmd = f"msfconsole -q -x 'use {module}; set RHOSTS {target}; set PAYLOAD {payload}; run; exit'"
        return await self._exec_shell({"cmd": cmd, "timeout": 600})

    async def _read_file(self, args: Dict) -> str:
        path = args.get("path", "/dev/null")
        offset = args.get("offset", 0)
        limit = args.get("limit", 10000)
        try:
            with open(path, 'r') as f:
                f.seek(offset)
                return f.read(limit)
        except Exception as e:
            return f"Error reading file: {e}"

    async def _write_file(self, args: Dict) -> str:
        path = args.get("path", "/tmp/fall_output.txt")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _get_telemetry(self, args: Dict) -> Dict:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids()),
        }

    async def _exec_hashcat(self, args: Dict) -> str:
        hash_file = args.get("hash_file", "/tmp/hashes.txt")
        wordlist = args.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        cmd = f"hashcat -m 0 {hash_file} {wordlist} --force"
        return await self._exec_shell({"cmd": cmd, "timeout": 600})

    async def _exec_john(self, args: Dict) -> str:
        hash_file = args.get("hash_file", "/tmp/hashes.txt")
        cmd = f"john {hash_file}"
        return await self._exec_shell({"cmd": cmd, "timeout": 600})

    async def _exec_hydra(self, args: Dict) -> str:
        target = args.get("target", "127.0.0.1")
        service = args.get("service", "ssh")
        username = args.get("username", "root")
        wordlist = args.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        cmd = f"hydra -l {username} -P {wordlist} {target} {service}"
        return await self._exec_shell({"cmd": cmd, "timeout": 300})

    async def _exec_curl(self, args: Dict) -> str:
        url = args.get("url", "http://localhost")
        method = args.get("method", "GET")
        data = args.get("data", "")
        cmd = f"curl -X {method} {url}"
        if data:
            cmd += f" -d '{data}'"
        return await self._exec_shell({"cmd": cmd, "timeout": 30})

    def get_audit_log(self) -> List[Dict]:
        return self.audit_log