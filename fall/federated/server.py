"""
Federated learning server for FALL.
Aggregates encrypted gradients from edge agents.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict
import copy
import logging

logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(
        self,
        global_model: nn.Module,
        min_clients: int = 10,
        dp_sigma: float = 1.0,
        dp_clip: float = 1.0,
    ):
        self.global_model = global_model
        self.min_clients = min_clients
        self.dp_sigma = dp_sigma
        self.dp_clip = dp_clip
        self.pending_updates: List[Dict] = []
        self.round = 0
        self.history: List[Dict] = []

    async def receive_update(self, client_id: str, encrypted_delta: bytes, round_num: int):
        """Receive an encrypted gradient update from a client."""
        # Decrypt and deserialize
        delta = self._decrypt(encrypted_delta)

        # Apply differential privacy
        delta = self._apply_dp(delta)

        self.pending_updates.append({
            "client_id": client_id,
            "delta": delta,
            "round": round_num,
        })

        # Aggregate if enough clients
        if len(self.pending_updates) >= self.min_clients:
            await self.aggregate()

    async def aggregate(self):
        """Aggregate pending updates using FedAvg."""
        if not self.pending_updates:
            return

        # Average deltas
        avg_delta = defaultdict(float)
        num_updates = len(self.pending_updates)
        for update in self.pending_updates:
            for name, param_delta in update["delta"].items():
                avg_delta[name] += param_delta / num_updates

        # Apply to global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in avg_delta:
                    param.data += avg_delta[name]

        self.round += 1
        logger.info(f"Round {self.round}: aggregated {num_updates} client updates")

        # Clear pending
        self.pending_updates = []

        # Publish new model
        await self._publish_model()

    async def _publish_model(self):
        """Publish the updated global model to all clients."""
        # Compress, sign, and push to CDN
        pass

    def _decrypt(self, encrypted_delta: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt client delta."""
        import pickle
        return pickle.loads(encrypted_delta)

    def _apply_dp(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to client update."""
        noised = {}
        for name, param in delta.items():
            # Clip
            norm = torch.norm(param)
            if norm > self.dp_clip:
                param = param * (self.dp_clip / norm)
            # Add noise
            noise = torch.randn_like(param) * self.dp_sigma * self.dp_clip
            noised[name] = param + noise
        return noised

    def get_model(self) -> nn.Module:
        return copy.deepcopy(self.global_model)