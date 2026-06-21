"""Tests for sandbox manager."""
import pytest
from fall.sandbox.manager import SandboxManager, SandboxConfig

class TestSandbox:
    @pytest.fixture
    def sandbox(self):
        config = SandboxConfig(network_enabled=False)
        return SandboxManager(config)
    
    def test_shell_execution(self, sandbox):
        result = asyncio.run(sandbox.execute("shell", {"cmd": "echo hello"}))
        assert result["success"] is True
    
    def test_python_execution(self, sandbox):
        result = asyncio.run(sandbox.execute("python", {"code": "print(1+1)"}))
        assert result["success"] is True
    
    def test_file_read_write(self, sandbox):
        write = asyncio.run(sandbox.execute("file_write", {
            "path": "/tmp/fall_test.txt",
            "content": "test content"
        }))
        assert write["success"] is True
        
        read = asyncio.run(sandbox.execute("file_read", {
            "path": "/tmp/fall_test.txt"
        }))
        assert "test content" in str(read["result"])