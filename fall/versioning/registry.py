"""
Model versioning registry for FALL.
Tracks model versions, checkpoints, and deployment history.
"""
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelVersion:
    version_id: str
    timestamp: float
    training_step: int
    commit_hash: str
    model_path: str
    metrics: Dict[str, float]
    status: str  # training, staging, production, archived
    deployed_at: Optional[float] = None
    rolled_back: bool = False

class ModelRegistry:
    def __init__(self, registry_path: str = "/models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.production_version: Optional[str] = None
        self._load()
    
    def _load(self):
        reg_file = self.registry_path / "registry.json"
        if reg_file.exists():
            with open(reg_file) as f:
                data = json.load(f)
                for v in data.get("versions", []):
                    mv = ModelVersion(**v)
                    self.versions[mv.version_id] = mv
                self.production_version = data.get("production_version")
    
    def _save(self):
        with open(self.registry_path / "registry.json", "w") as f:
            json.dump({
                "versions": [v.__dict__ for v in self.versions.values()],
                "production_version": self.production_version,
            }, f, indent=2)
    
    def register(
        self,
        training_step: int,
        model_path: str,
        metrics: Dict[str, float],
        commit_hash: str = "unknown",
    ) -> str:
        version_id = f"v{int(time.time())}_{training_step:07d}_{hashlib.sha256(commit_hash.encode()).hexdigest()[:8]}"
        
        self.versions[version_id] = ModelVersion(
            version_id=version_id,
            timestamp=time.time(),
            training_step=training_step,
            commit_hash=commit_hash,
            model_path=model_path,
            metrics=metrics,
            status="training",
        )
        self._save()
        return version_id
    
    def promote_to_staging(self, version_id: str):
        if version_id in self.versions:
            self.versions[version_id].status = "staging"
            self._save()
    
    def promote_to_production(self, version_id: str):
        if version_id in self.versions:
            if self.production_version:
                self.versions[self.production_version].status = "archived"
            self.versions[version_id].status = "production"
            self.versions[version_id].deployed_at = time.time()
            self.production_version = version_id
            self._save()
    
    def rollback(self, to_version_id: str):
        if to_version_id in self.versions:
            if self.production_version:
                self.versions[self.production_version].rolled_back = True
                self.versions[self.production_version].status = "archived"
            self.versions[to_version_id].status = "production"
            self.production_version = to_version_id
            self._save()
    
    def get_production_model_path(self) -> Optional[str]:
        if self.production_version:
            return self.versions[self.production_version].model_path
        return None
    
    def list_versions(self) -> List[Dict]:
        return [
            {"id": v.version_id, "step": v.training_step, "status": v.status}
            for v in sorted(self.versions.values(), key=lambda x: x.timestamp, reverse=True)
        ]