"""
Automated documentation generator for FALL.
Extracts docstrings and generates API docs.
"""
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re

class DocGenerator:
    def __init__(self, source_dir: str = "fall"):
        self.source_dir = Path(source_dir)
        self.modules: Dict[str, Dict] = {}
    
    def parse_all(self):
        """Parse all Python files in the source directory."""
        for py_file in self.source_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            self._parse_file(py_file)
    
    def _parse_file(self, filepath: Path):
        with open(filepath, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return
        
        module_name = str(filepath.relative_to(self.source_dir)).replace("/", ".").replace(".py", "")
        module_info = {
            "classes": [],
            "functions": [],
            "docstring": ast.get_docstring(tree) or "",
        }
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                module_info["classes"].append(self._parse_class(node))
            elif isinstance(node, ast.FunctionDef):
                module_info["functions"].append(self._parse_function(node))
        
        self.modules[module_name] = module_info
    
    def _parse_class(self, node: ast.ClassDef) -> Dict:
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "methods": [
                self._parse_function(m) for m in node.body
                if isinstance(m, ast.FunctionDef)
            ],
            "bases": [self._get_name(b) for b in node.bases],
        }
    
    def _parse_function(self, node: ast.FunctionDef) -> Dict:
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "args": [arg.arg for arg in node.args.args],
            "returns": self._get_return_annotation(node),
        }
    
    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _get_return_annotation(self, node):
        if node.returns:
            return ast.unparse(node.returns)
        return "None"
    
    def generate_markdown(self) -> str:
        """Generate Markdown documentation."""
        lines = [
            "# FALL API Documentation",
            "",
            f"**Version:** 1.0.0",
            f"**Total Modules:** {len(self.modules)}",
            "",
            "---",
            "",
        ]
        
        for module_name, module_info in sorted(self.modules.items()):
            lines.append(f"## Module `{module_name}`")
            if module_info["docstring"]:
                lines.append(f"\n{module_info['docstring']}\n")
            
            for cls in module_info["classes"]:
                lines.append(f"### Class `{cls['name']}`")
                if cls["docstring"]:
                    lines.append(f"\n{cls['docstring']}\n")
                if cls["bases"]:
                    lines.append(f"**Inherits from:** {', '.join(cls['bases'])}\n")
                
                for method in cls["methods"]:
                    args_str = ", ".join(method["args"])
                    lines.append(f"#### `{method['name']}({args_str}) -> {method['returns']}`")
                    if method["docstring"]:
                        lines.append(f"\n{method['docstring']}\n")
            
            for func in module_info["functions"]:
                args_str = ", ".join(func["args"])
                lines.append(f"### `{func['name']}({args_str}) -> {func['returns']}`")
                if func["docstring"]:
                    lines.append(f"\n{func['docstring']}\n")
            
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def generate_html(self) -> str:
        """Generate HTML documentation."""
        markdown = self.generate_markdown()
        # Simple conversion (use mistune or markdown in production)
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>FALL API Documentation</title>
    <style>
        body {{ font-family: system-ui; max-width: 900px; margin: auto; padding: 20px; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f0f0f0; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <pre>{markdown}</pre>
</body>
</html>
"""


def main():
    generator = DocGenerator()
    generator.parse_all()
    
    md = generator.generate_markdown()
    with open("docs/API.md", "w") as f:
        f.write(md)
    
    html = generator.generate_html()
    with open("docs/API.html", "w") as f:
        f.write(html)
    
    print(f"Documentation generated: {len(generator.modules)} modules")


if __name__ == "__main__":
    main()