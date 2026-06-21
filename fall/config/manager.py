"""
Configuration management for FALL.
Supports YAML, env vars, and CLI overrides.
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class FALLGlobalConfig:
    # Model
    model: str = "fall_5t"
    model_path: str = "/models/fall_5t"
    tokenizer_path: str = "/models/tokenizer.json"
    
    # Training
    train_batch_size: int = 4096
    train_seq_len: int = 524288
    train_steps: int = 3_750_000
    learning_rate: float = 1.5e-4
    
    # Inference
    inference_port: int = 8000
    inference_host: str = "0.0.0.0"
    inference_tensor_parallel: int = 16
    
    # Data
    data_path: str = "/data"
    cache_dir: str = "/cache"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "/var/log/fall/fall.log"
    
    # API
    api_keys_file: str = "/etc/fall/api_keys.yaml"
    rate_limit: int = 1000
    
    # Agent
    agent_enabled: bool = True
    agent_interval: float = 2.0
    
    # Cluster
    world_size: int = 8192
    master_addr: str = "localhost"
    master_port: int = 29500

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config = FALLGlobalConfig()
        
        # Load from defaults
        self._load_defaults()
        
        # Load from YAML
        if config_path:
            self._load_yaml(config_path)
        
        # Override from environment
        self._load_env()
    
    def _load_defaults(self):
        pass
    
    def _load_yaml(self, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _load_env(self):
        for key in self.config.__dataclass_fields__:
            env_key = f"FALL_{key.upper()}"
            if env_key in os.environ:
                setattr(self.config, key, os.environ[env_key])
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config.__dict__