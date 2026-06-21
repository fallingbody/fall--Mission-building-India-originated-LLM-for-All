"""
Multi-modal input processor for FALL.
Handles images, PDFs, code diffs, and structured data.
"""
import base64
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import re

@dataclass
class MultiModalInput:
    type: str  # text, image, pdf, code_diff, json, csv
    content: Any
    metadata: Dict[str, Any] = None

class MultiModalProcessor:
    def __init__(self):
        self.processors = {
            "text": self._process_text,
            "image": self._process_image,
            "pdf": self._process_pdf,
            "code_diff": self._process_code_diff,
            "json": self._process_json,
            "csv": self._process_csv,
            "log": self._process_log,
        }
    
    def process(self, inputs: List[MultiModalInput]) -> str:
        """Process multiple inputs into a unified text representation."""
        processed = []
        
        for inp in inputs:
            if inp.type in self.processors:
                result = self.processors[inp.type](inp.content, inp.metadata or {})
                processed.append(result)
            else:
                processed.append(f"[Unknown format: {inp.type}]\n{str(inp.content)[:500]}")
        
        return "\n\n".join(processed)
    
    def _process_text(self, content: str, metadata: Dict) -> str:
        return content
    
    def _process_image(self, content: Union[str, bytes], metadata: Dict) -> str:
        """Convert image to a text description placeholder."""
        if isinstance(content, bytes):
            content = base64.b64encode(content).decode()
        
        return f"""[IMAGE]
Description: {metadata.get('description', 'No description provided')}
Dimensions: {metadata.get('width', '?')}x{metadata.get('height', '?')}
Format: {metadata.get('format', 'unknown')}
[END IMAGE]"""
    
    def _process_pdf(self, content: Union[str, bytes], metadata: Dict) -> str:
        """Extract text from PDF."""
        # In production, use PyMuPDF or similar
        return f"[PDF: {metadata.get('filename', 'unknown')}]\n{str(content)[:10000]}"
    
    def _process_code_diff(self, content: str, metadata: Dict) -> str:
        """Format a code diff for the model."""
        lines = content.split('\n')
        formatted = []
        for line in lines:
            if line.startswith('+'):
                formatted.append(f"[ADDED]   {line[1:]}")
            elif line.startswith('-'):
                formatted.append(f"[REMOVED] {line[1:]}")
            elif line.startswith('@@'):
                formatted.append(f"[CONTEXT] {line}")
            else:
                formatted.append(f"          {line}")
        
        return "Code Diff:\n```diff\n" + "\n".join(formatted) + "\n```"
    
    def _process_json(self, content: Union[str, Dict], metadata: Dict) -> str:
        """Format JSON for the model."""
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        return f"```json\n{content}\n```"
    
    def _process_csv(self, content: str, metadata: Dict) -> str:
        """Format CSV as a markdown table."""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return content
        
        headers = lines[0].split(',')
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for line in lines[1:]:
            values = line.split(',')
            table += "| " + " | ".join(values[:len(headers)]) + " |\n"
        
        return table
    
    def _process_log(self, content: str, metadata: Dict) -> str:
        """Format log files with line numbers."""
        lines = content.split('\n')
        formatted = [f"[LOG: {metadata.get('filename', 'unknown')}]"]
        for i, line in enumerate(lines[:100], 1):
            formatted.append(f"{i:4d} | {line}")
        if len(lines) > 100:
            formatted.append(f"... ({len(lines) - 100} more lines)")
        return '\n'.join(formatted)