"""
Multi-tenant request isolation for FALL API.
Ensures strict separation between API key holders.
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Tenant:
    api_key_hash: str
    role: str
    quota_total: int
    quota_used: int
    rate_limit_rpm: int
    request_history: List[float] = field(default_factory=list)
    active_sandbox_id: Optional[str] = None
    
    def can_request(self) -> bool:
        """Check if tenant is within rate limits."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_history = [t for t in self.request_history if now - t < 60]
        return len(self.request_history) < self.rate_limit_rpm
    
    def record_request(self):
        self.request_history.append(time.time())
        self.quota_used += 1


class TenantIsolator:
    """Manages tenant isolation and resource allocation."""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.sandbox_assignments: Dict[str, str] = {}  # sandbox_id -> api_key_hash
        self.lock = asyncio.Lock()
    
    async def register_tenant(
        self,
        api_key: str,
        role: str = "user",
        quota: int = 10000,
        rate_limit: int = 100,
    ) -> str:
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        async with self.lock:
            self.tenants[key_hash] = Tenant(
                api_key_hash=key_hash,
                role=role,
                quota_total=quota,
                quota_used=0,
                rate_limit_rpm=rate_limit,
            )
        return key_hash
    
    async def validate_request(self, api_key: str) -> Optional[str]:
        """Validate an API key and return tenant hash if allowed."""
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        async with self.lock:
            tenant = self.tenants.get(key_hash)
            if tenant is None:
                return None
            if not tenant.can_request():
                return None
            if tenant.quota_used >= tenant.quota_total:
                return None
            tenant.record_request()
            return key_hash
    
    async def get_sandbox(self, tenant_hash: str) -> str:
        """Get or create an isolated sandbox for a tenant."""
        async with self.lock:
            tenant = self.tenants.get(tenant_hash)
            if tenant and tenant.active_sandbox_id:
                return tenant.active_sandbox_id
            
            sandbox_id = f"sandbox_{tenant_hash[:8]}_{uuid.uuid4().hex[:8]}"
            if tenant:
                tenant.active_sandbox_id = sandbox_id
            self.sandbox_assignments[sandbox_id] = tenant_hash
            return sandbox_id
    
    async def cleanup_tenant(self, tenant_hash: str):
        """Clean up tenant resources."""
        async with self.lock:
            tenant = self.tenants.get(tenant_hash)
            if tenant and tenant.active_sandbox_id:
                self.sandbox_assignments.pop(tenant.active_sandbox_id, None)
                tenant.active_sandbox_id = None