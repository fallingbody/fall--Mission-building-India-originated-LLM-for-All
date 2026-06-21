"""
Knowledge graph integration for FALL.
Persistent, queryable semantic memory for long-term reasoning.
"""
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Entity:
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None

@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    def __init__(self, persist_path: str = "/data/knowledge_graph"):
        self.persist_path = persist_path
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self._load()
    
    def _load(self):
        try:
            import os
            entities_file = f"{self.persist_path}/entities.json"
            relations_file = f"{self.persist_path}/relations.json"
            if os.path.exists(entities_file):
                with open(entities_file) as f:
                    for line in f:
                        e = Entity(**json.loads(line))
                        self.entities[e.id] = e
                        self.index[e.type].add(e.id)
            if os.path.exists(relations_file):
                with open(relations_file) as f:
                    for line in f:
                        self.relations.append(Relation(**json.loads(line)))
        except Exception:
            pass
    
    def _save(self):
        import os
        os.makedirs(self.persist_path, exist_ok=True)
        with open(f"{self.persist_path}/entities.json", 'w') as f:
            for e in self.entities.values():
                f.write(json.dumps(e.__dict__) + '\n')
        with open(f"{self.persist_path}/relations.json", 'w') as f:
            for r in self.relations:
                f.write(json.dumps(r.__dict__) + '\n')
    
    def add_entity(self, entity: Entity) -> str:
        if not entity.id:
            entity.id = hashlib.sha256(
                f"{entity.type}:{json.dumps(entity.properties)}".encode()
            ).hexdigest()[:16]
        
        self.entities[entity.id] = entity
        self.index[entity.type].add(entity.id)
        self._save()
        return entity.id
    
    def add_relation(self, source_id: str, target_id: str, rel_type: str, weight: float = 1.0):
        if source_id not in self.entities or target_id not in self.entities:
            return
        
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            weight=weight,
        )
        self.relations.append(relation)
        self._save()
    
    def query(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        relation_type: Optional[str] = None,
    ) -> List[Entity]:
        """Query entities by type, properties, and relations."""
        candidate_ids = set()
        
        if entity_type:
            candidate_ids = self.index.get(entity_type, set())
        else:
            candidate_ids = set(self.entities.keys())
        
        if properties:
            candidate_ids = {
                eid for eid in candidate_ids
                if all(
                    self.entities[eid].properties.get(k) == v
                    for k, v in properties.items()
                )
            }
        
        if relation_type:
            related_ids = set()
            for r in self.relations:
                if r.type == relation_type:
                    if r.source_id in candidate_ids:
                        related_ids.add(r.target_id)
                    if r.target_id in candidate_ids:
                        related_ids.add(r.source_id)
            candidate_ids = related_ids
        
        return [self.entities[eid] for eid in candidate_ids]
    
    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get the neighborhood of an entity up to a given depth."""
        visited = set()
        result = {"entity": None, "relations": [], "neighbors": []}
        
        if entity_id not in self.entities:
            return result
        
        result["entity"] = self.entities[entity_id]
        queue = [(entity_id, 0)]
        visited.add(entity_id)
        
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
            
            for rel in self.relations:
                neighbor_id = None
                if rel.source_id == current_id:
                    neighbor_id = rel.target_id
                elif rel.target_id == current_id:
                    neighbor_id = rel.source_id
                
                if neighbor_id and neighbor_id not in visited:
                    if relation_types is None or rel.type in relation_types:
                        visited.add(neighbor_id)
                        result["relations"].append(rel)
                        result["neighbors"].append(self.entities[neighbor_id])
                        queue.append((neighbor_id, current_depth + 1))
        
        return result