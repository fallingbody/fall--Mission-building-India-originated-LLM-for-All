"""
Synthetic data generation pipeline for FALL.
Generates high-quality training data via self-play, backtranslation, and augmentation.
"""
import asyncio
import json
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class SyntheticTask:
    instruction: str
    response: str
    tool_calls: List[Dict]
    quality_score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class SyntheticDataGenerator:
    def __init__(self, model_api, sandbox, quality_threshold: float = 0.7):
        self.model = model_api
        self.sandbox = sandbox
        self.quality_threshold = quality_threshold
        self.generated_tasks: List[SyntheticTask] = []
        self.task_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        return {
            "code_generation": [
                "Write a {language} function that {task_description}",
                "Implement a {design_pattern} pattern for {use_case}",
                "Create a {language} script to automate {process}",
            ],
            "vulnerability_discovery": [
                "Find the vulnerability in this {language} code:\n```\n{code}\n```",
                "Explain why this code is insecure:\n```\n{code}\n```",
                "Identify all OWASP Top 10 issues in this application code",
            ],
            "exploit_development": [
                "Write a proof-of-concept exploit for {cve_id}",
                "Develop a working exploit for the vulnerability described",
                "Create a Metasploit module for {vulnerability}",
            ],
            "security_hardening": [
                "Audit this server configuration for security issues",
                "Write a security policy for {system_type}",
                "Generate CIS benchmark compliance report for this system",
            ],
            "reasoning": [
                "Analyze this log file and identify the attack vector",
                "Trace this exploit chain and explain each step",
                "Compare two security approaches for {scenario}",
            ],
        }
    
    async def generate_batch(
        self,
        num_samples: int = 1000,
        categories: Optional[List[str]] = None,
    ) -> List[SyntheticTask]:
        """Generate a batch of synthetic training data."""
        if categories is None:
            categories = list(self.task_templates.keys())
        
        tasks = []
        for _ in range(num_samples):
            category = random.choice(categories)
            template = random.choice(self.task_templates[category])
            
            # Generate instruction
            instruction = self._fill_template(template)
            
            # Generate response using self-play
            try:
                response = await self._generate_response(instruction, category)
                
                # Verify quality
                quality = await self._evaluate_quality(instruction, response)
                
                if quality >= self.quality_threshold:
                    task = SyntheticTask(
                        instruction=instruction,
                        response=response.get("text", ""),
                        tool_calls=response.get("tool_calls", []),
                        quality_score=quality,
                        source=f"synthetic_{category}",
                    )
                    tasks.append(task)
                    self.generated_tasks.append(task)
            except Exception:
                pass
        
        return tasks
    
    def _fill_template(self, template: str) -> str:
        """Fill template variables with realistic values."""
        variables = {
            "language": random.choice(["Python", "Rust", "Go", "JavaScript", "C"]),
            "task_description": random.choice([
                "parse JSON files", "sort a list of objects",
                "connect to a database", "encrypt a file",
                "validate input", "send an HTTP request",
            ]),
            "design_pattern": random.choice(["Singleton", "Factory", "Observer", "Strategy"]),
            "use_case": random.choice(["logging", "caching", "authentication", "data processing"]),
            "process": random.choice(["backups", "log rotation", "data migration", "monitoring"]),
            "code": "def vulnerable_function(user_input):\n    eval(user_input)  # Security risk",
            "cve_id": f"CVE-{random.randint(2017, 2024)}-{random.randint(1000, 99999)}",
            "vulnerability": random.choice(["buffer overflow", "SQL injection", "XSS", "CSRF"]),
            "system_type": random.choice(["web server", "database", "container", "network"]),
            "scenario": random.choice(["cloud deployment", "on-premise", "hybrid", "IoT"]),
        }
        
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    async def _generate_response(self, instruction: str, category: str) -> Dict:
        """Generate a response using the model."""
        prompt = f"""
        <|system|>
        You are generating high-quality training data for category: {category}.
        Provide a thorough, accurate, and well-formatted response.
        Include code examples where appropriate.
        If the task requires action, use <|tool_call|> blocks.
        </|system|>
        
        <|user|>
        {instruction}
        </|user|>
        
        <|assistant|>
        """
        return await self.model.generate(prompt, max_tokens=2048)
    
    async def _evaluate_quality(self, instruction: str, response: str) -> float:
        """Evaluate the quality of a generated response."""
        scores = []
        
        # Length check
        if len(response) > 50:
            scores.append(0.3)
        
        # Code presence (for code-related tasks)
        if "code" in instruction.lower() or "write" in instruction.lower():
            if "```" in response:
                scores.append(0.3)
        
        # Tool call presence (for action tasks)
        if "exploit" in instruction.lower() or "scan" in instruction.lower():
            if "<|tool_call|>" in response:
                scores.append(0.3)
        
        # Coherence (simple heuristic)
        if len(response.split()) > 20:
            scores.append(0.3)
        
        return sum(scores)
    
    async def generate_code_self_play(self, num_samples: int = 500) -> List[SyntheticTask]:
        """Generate code via self-play: write, execute, fix, repeat."""
        tasks = []
        
        for _ in range(num_samples):
            # Step 1: Generate code
            spec = self._fill_template("Write a {language} function that {task_description}")
            response = await self.model.generate(spec, max_tokens=1024)
            code = self._extract_code(response.get("text", ""))
            
            if not code:
                continue
            
            # Step 2: Execute in sandbox
            exec_result = await self.sandbox.execute("python", {"code": code})
            
            # Step 3: If failed, generate fix
            if not exec_result.get("success", False):
                error_msg = exec_result.get("result", "")
                fix_prompt = f"Fix this code error:\n```\n{code}\n```\nError: {error_msg}"
                fix_response = await self.model.generate(fix_prompt, max_tokens=1024)
                fixed_code = self._extract_code(fix_response.get("text", ""))
                
                if fixed_code:
                    # Re-execute to verify
                    fix_result = await self.sandbox.execute("python", {"code": fixed_code})
                    if fix_result.get("success", False):
                        tasks.append(SyntheticTask(
                            instruction=spec,
                            response=fix_response.get("text", ""),
                            tool_calls=[],
                            quality_score=0.9,
                            source="synthetic_code_self_play",
                            metadata={"original_code": code, "error": error_msg},
                        ))
            else:
                tasks.append(SyntheticTask(
                    instruction=spec,
                    response=response.get("text", ""),
                    tool_calls=[],
                    quality_score=0.85,
                    source="synthetic_code_self_play",
                ))
        
        return tasks
    
    def _extract_code(self, text: str) -> Optional[str]:
        match = re.search(r'```(?:python|rust|go|javascript|c)?\n(.*?)```', text, re.DOTALL)
        return match.group(1) if match else None
    
    def export_dataset(self, path: str = "synthetic_data.jsonl"):
        with open(path, 'w') as f:
            for task in self.generated_tasks:
                f.write(json.dumps(task.__dict__) + '\n')