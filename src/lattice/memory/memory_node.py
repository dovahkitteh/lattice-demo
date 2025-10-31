import uuid
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEMORY NODE ARCHITECTURE - PHASE 1 FOUNDATION
# ---------------------------------------------------------------------------

class MemoryNodeType(Enum):
    """Types of memory nodes in the lattice system"""
    SMG = "smg"                    # Single Memory Generation
    DUAL_AFFECT = "dual_affect"    # Dual-channel affect memory
    RECURSION = "recursion"        # Recursion cycle memory
    ECHO = "echo"                  # Echo relationship memory
    COMPRESSED = "compressed"      # Compressed conversation memory
    CUSTOM = "custom"              # Custom memory type

class MemoryLifecycleState(Enum):
    """Lifecycle states for memory evolution"""
    RAW = "raw"                    # Newly created memory
    ECHOED = "echoed"              # Memory that has been accessed
    CRYSTALLIZED = "crystallized"  # High-importance memory
    ARCHIVED = "archived"          # Old but preserved memory
    COMPRESSED = "compressed"      # Summarized memory
    SHADOW = "shadow"              # Integrated into shadow system

class MemoryOrigin(Enum):
    """Origins of memory creation"""
    EXTERNAL_USER = "external_user"
    DUAL_CHANNEL = "dual_channel"
    RECURSION_CYCLE = "recursion_cycle"
    SHADOW_INTEGRATION = "shadow_integration"
    DAEMON_STATEMENT = "daemon_statement"
    COMPRESSION = "compression"
    ECHO_STRENGTHENING = "echo_strengthening"

@dataclass
class MemoryNode:
    """
    Unified memory node that can represent all memory types in the lattice system.
    
    This class maintains complete backwards compatibility with existing storage formats
    while providing a foundation for enhanced memory capabilities.
    """
    
    # Core identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: MemoryNodeType = MemoryNodeType.SMG
    
    # Content and metadata
    content: str = ""
    synopsis: str = ""
    reflection: Optional[str] = None
    
    # Temporal information
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    origin: MemoryOrigin = MemoryOrigin.EXTERNAL_USER
    
    # Emotional data (28-dimensional GoEmotions vectors)
    user_affect_vector: Optional[List[float]] = None
    self_affect_vector: Optional[List[float]] = None
    primary_affect_vector: Optional[List[float]] = None  # For single-affect memories
    
    # Lifecycle and access tracking
    lifecycle_state: MemoryLifecycleState = MemoryLifecycleState.RAW
    echo_count: int = 0
    echo_strength: float = 0.0
    last_echo_timestamp: Optional[datetime] = None
    last_echo_query: Optional[str] = None
    
    # Relationship and graph data
    parent_node_id: Optional[str] = None
    child_node_ids: List[str] = field(default_factory=list)
    related_node_ids: List[str] = field(default_factory=list)
    
    # Semantic embedding (for ChromaDB storage)
    semantic_embedding: Optional[List[float]] = None
    
    # Enhanced metadata for future capabilities
    importance_score: float = 0.0
    emotional_significance: float = 0.0
    personality_influence: float = 0.0
    
    # Backwards compatibility fields
    legacy_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure affect vectors are correct length if provided
        if self.user_affect_vector and len(self.user_affect_vector) != 28:
            logger.warning(f"User affect vector length {len(self.user_affect_vector)} != 28, padding/truncating")
            self.user_affect_vector = self._normalize_affect_vector(self.user_affect_vector)
            
        if self.self_affect_vector and len(self.self_affect_vector) != 28:
            logger.warning(f"Self affect vector length {len(self.self_affect_vector)} != 28, padding/truncating")
            self.self_affect_vector = self._normalize_affect_vector(self.self_affect_vector)
            
        if self.primary_affect_vector and len(self.primary_affect_vector) != 28:
            logger.warning(f"Primary affect vector length {len(self.primary_affect_vector)} != 28, padding/truncating")
            self.primary_affect_vector = self._normalize_affect_vector(self.primary_affect_vector)
        
        # Calculate derived metrics
        self._calculate_emotional_significance()
        self._validate_node_type()
    
    def _normalize_affect_vector(self, vector: List[float]) -> List[float]:
        """Normalize affect vector to 28 dimensions"""
        if len(vector) == 28:
            return vector
        elif len(vector) < 28:
            return vector + [0.0] * (28 - len(vector))
        else:
            return vector[:28]
    
    def _calculate_emotional_significance(self):
        """Calculate emotional significance from affect vectors"""
        total_magnitude = 0.0
        
        if self.user_affect_vector:
            total_magnitude += sum(abs(x) for x in self.user_affect_vector)
        
        if self.self_affect_vector:
            total_magnitude += sum(abs(x) for x in self.self_affect_vector)
            
        if self.primary_affect_vector:
            total_magnitude += sum(abs(x) for x in self.primary_affect_vector)
            
        self.emotional_significance = total_magnitude
    
    def _validate_node_type(self):
        """Validate node type based on available data"""
        if self.user_affect_vector and self.self_affect_vector:
            if self.node_type == MemoryNodeType.SMG:
                self.node_type = MemoryNodeType.DUAL_AFFECT
        elif self.primary_affect_vector and self.node_type == MemoryNodeType.DUAL_AFFECT:
            self.node_type = MemoryNodeType.SMG
    
    # ---------------------------------------------------------------------------
    # BACKWARDS COMPATIBILITY METHODS
    # ---------------------------------------------------------------------------
    
    def to_legacy_smg_format(self) -> Dict[str, Any]:
        """Convert to legacy SMG storage format for backwards compatibility"""
        affect_vec = self.primary_affect_vector or [0.0] * 28
        
        return {
            "node_id": self.node_id,
            "msg": self.content,
            "affect_vec": affect_vec,
            "synopsis": self.synopsis,
            "reflection": self.reflection,
            "origin": self.origin.value if isinstance(self.origin, MemoryOrigin) else str(self.origin)
        }
    
    def to_legacy_dual_affect_format(self) -> Dict[str, Any]:
        """Convert to legacy dual-affect storage format"""
        user_affect = self.user_affect_vector or [0.0] * 28
        self_affect = self.self_affect_vector or [0.0] * 28
        
        return {
            "node_id": self.node_id,
            "msg": self.content,
            "user_affect": user_affect,
            "self_affect": self_affect,
            "synopsis": self.synopsis,
            "reflection": self.reflection,
            "origin": self.origin.value if isinstance(self.origin, MemoryOrigin) else str(self.origin)
        }
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format"""
        metadata = {
            "node_id": self.node_id,
            "synopsis": self.synopsis,
            "origin": self.origin.value if isinstance(self.origin, MemoryOrigin) else str(self.origin),
            "timestamp": self.timestamp.isoformat(),
            "has_reflection": self.reflection is not None,
            "node_type": self.node_type.value,
            "lifecycle_state": self.lifecycle_state.value,
            "echo_count": self.echo_count,
            "echo_strength": self.echo_strength,
            "importance_score": self.importance_score,
            "emotional_significance": self.emotional_significance,
        }
        
        # Add reflection if present
        if self.reflection:
            metadata["reflection"] = self.reflection
        
        # Add affect magnitudes
        if self.user_affect_vector:
            metadata["user_affect_magnitude"] = sum(abs(x) for x in self.user_affect_vector)
        if self.self_affect_vector:
            metadata["self_affect_magnitude"] = sum(abs(x) for x in self.self_affect_vector)
        if self.primary_affect_vector:
            metadata["affect_magnitude"] = sum(abs(x) for x in self.primary_affect_vector)
        
        # Mark dual-channel if applicable
        if self.user_affect_vector and self.self_affect_vector:
            metadata["dual_channel"] = True
        
        # Add echo information
        if self.last_echo_timestamp:
            metadata["last_echo"] = self.last_echo_timestamp.isoformat()
        if self.last_echo_query:
            metadata["last_echo_query"] = self.last_echo_query
        
        # Add legacy metadata, but serialize complex structures
        for key, value in self.legacy_metadata.items():
            if isinstance(value, (dict, list)):
                # Serialize complex structures as JSON strings
                import json
                metadata[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # ChromaDB accepts these types directly
                metadata[key] = value
            else:
                # Convert other types to strings
                metadata[key] = str(value)
        
        return metadata
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties"""
        properties = {
            "id": self.node_id,
            "content": self.content,
            "synopsis": self.synopsis,
            "reflection": self.reflection,
            "origin": self.origin.value if isinstance(self.origin, MemoryOrigin) else str(self.origin),
            "timestamp": self.timestamp.isoformat(),
            "node_type": self.node_type.value,
            "lifecycle_state": self.lifecycle_state.value,
            "echo_count": self.echo_count,
            "echo_strength": self.echo_strength,
            "importance_score": self.importance_score,
            "emotional_significance": self.emotional_significance,
        }
        
        # Add affect data
        if self.user_affect_vector:
            properties["user_affect_magnitude"] = sum(abs(x) for x in self.user_affect_vector)
            properties["user_affect_vector"] = self.user_affect_vector
        if self.self_affect_vector:
            properties["self_affect_magnitude"] = sum(abs(x) for x in self.self_affect_vector)
            properties["self_affect_vector"] = self.self_affect_vector
        if self.primary_affect_vector:
            properties["affect_magnitude"] = sum(abs(x) for x in self.primary_affect_vector)
            properties["affect_vector"] = self.primary_affect_vector
        
        # Add relationship data
        if self.parent_node_id:
            properties["parent_node_id"] = self.parent_node_id
        if self.child_node_ids:
            properties["child_node_ids"] = self.child_node_ids
        if self.related_node_ids:
            properties["related_node_ids"] = self.related_node_ids
        
        return properties
    
    def get_neo4j_label(self) -> str:
        """Get appropriate Neo4j label for backwards compatibility"""
        if self.node_type == MemoryNodeType.DUAL_AFFECT:
            return "DualMemoryNode"
        elif self.node_type == MemoryNodeType.ECHO:
            return "EchoNode"
        else:
            return "MemoryNode"
    
    # ---------------------------------------------------------------------------
    # FACTORY METHODS FOR BACKWARDS COMPATIBILITY
    # ---------------------------------------------------------------------------
    
    @classmethod
    def from_legacy_smg(cls, node_id: str, content: str, affect_vec: List[float], 
                       synopsis: str, reflection: Optional[str] = None, 
                       origin: str = "external_user") -> 'MemoryNode':
        """Create MemoryNode from legacy SMG format"""
        return cls(
            node_id=node_id,
            node_type=MemoryNodeType.SMG,
            content=content,
            synopsis=synopsis,
            reflection=reflection,
            primary_affect_vector=affect_vec,
            origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.EXTERNAL_USER
        )
    
    @classmethod
    def from_legacy_dual_affect(cls, node_id: str, content: str, 
                               user_affect: List[float], self_affect: List[float],
                               synopsis: str, reflection: Optional[str] = None,
                               origin: str = "dual_channel") -> 'MemoryNode':
        """Create MemoryNode from legacy dual-affect format"""
        return cls(
            node_id=node_id,
            node_type=MemoryNodeType.DUAL_AFFECT,
            content=content,
            synopsis=synopsis,
            reflection=reflection,
            user_affect_vector=user_affect,
            self_affect_vector=self_affect,
            origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.DUAL_CHANNEL
        )
    
    @classmethod
    def from_chroma_result(cls, document: str, metadata: Dict[str, Any]) -> 'MemoryNode':
        """Create MemoryNode from ChromaDB query result"""
        node = cls(
            node_id=metadata.get("node_id", str(uuid.uuid4())),
            content=document,
            synopsis=metadata.get("synopsis", ""),
            reflection=metadata.get("reflection"),
            echo_count=metadata.get("echo_count", 0),
            echo_strength=metadata.get("echo_strength", 0.0),
            importance_score=metadata.get("importance_score", 0.0),
            emotional_significance=metadata.get("emotional_significance", 0.0),
        )
        
        # Parse timestamp
        if "timestamp" in metadata:
            try:
                node.timestamp = datetime.fromisoformat(metadata["timestamp"])
            except ValueError:
                pass
        
        # Parse enums
        if "node_type" in metadata:
            try:
                node.node_type = MemoryNodeType(metadata["node_type"])
            except ValueError:
                pass
        
        if "lifecycle_state" in metadata:
            try:
                node.lifecycle_state = MemoryLifecycleState(metadata["lifecycle_state"])
            except ValueError:
                pass
        
        if "origin" in metadata:
            try:
                node.origin = MemoryOrigin(metadata["origin"])
            except ValueError:
                pass
        
        # Parse echo information
        if "last_echo" in metadata:
            try:
                node.last_echo_timestamp = datetime.fromisoformat(metadata["last_echo"])
            except ValueError:
                pass
        
        if "last_echo_query" in metadata:
            node.last_echo_query = metadata["last_echo_query"]
        
        # Store unrecognized metadata
        recognized_keys = {
            "node_id", "synopsis", "reflection", "timestamp", "node_type", 
            "lifecycle_state", "origin", "echo_count", "echo_strength",
            "importance_score", "emotional_significance", "last_echo", 
            "last_echo_query", "has_reflection", "dual_channel",
            "user_affect_magnitude", "self_affect_magnitude", "affect_magnitude",
            "session_id", "turn_id"
        }
        
        node.legacy_metadata = {k: v for k, v in metadata.items() if k not in recognized_keys}
        
        return node
    
    # ---------------------------------------------------------------------------
    # ENHANCED MEMORY CAPABILITIES
    # ---------------------------------------------------------------------------
    
    def update_echo_access(self, query_text: str, query_affect: List[float]):
        """Update echo information when memory is accessed"""
        self.echo_count += 1
        self.last_echo_timestamp = datetime.now(timezone.utc)
        self.last_echo_query = query_text[:100] if query_text else None
        
        # Update echo strength based on query affect
        if query_affect:
            query_magnitude = sum(abs(x) for x in query_affect)
            self.echo_strength += query_magnitude
        
        # Update lifecycle state based on echo patterns
        if self.lifecycle_state == MemoryLifecycleState.RAW and self.echo_count >= 2:
            self.lifecycle_state = MemoryLifecycleState.ECHOED
        elif self.lifecycle_state == MemoryLifecycleState.ECHOED and self.echo_strength > 10.0:
            self.lifecycle_state = MemoryLifecycleState.CRYSTALLIZED
    
    def calculate_retrieval_score(self, query_affect: List[float], 
                                 personality_state: Optional[Dict[str, float]] = None) -> float:
        """Calculate retrieval relevance score for enhanced search"""
        base_score = self.importance_score
        
        # Emotional resonance score
        if query_affect:
            emotional_match = 0.0
            if self.user_affect_vector:
                emotional_match += sum(a * b for a, b in zip(query_affect, self.user_affect_vector))
            if self.self_affect_vector:
                emotional_match += sum(a * b for a, b in zip(query_affect, self.self_affect_vector))
            if self.primary_affect_vector:
                emotional_match += sum(a * b for a, b in zip(query_affect, self.primary_affect_vector))
            
            base_score += emotional_match * 0.3
        
        # Lifecycle importance bonus
        lifecycle_bonuses = {
            MemoryLifecycleState.RAW: 0.0,
            MemoryLifecycleState.ECHOED: 0.1,
            MemoryLifecycleState.CRYSTALLIZED: 0.3,
            MemoryLifecycleState.ARCHIVED: -0.1,
            MemoryLifecycleState.COMPRESSED: -0.2,
            MemoryLifecycleState.SHADOW: 0.2,
        }
        base_score += lifecycle_bonuses.get(self.lifecycle_state, 0.0)
        
        # Echo frequency bonus
        base_score += min(self.echo_count * 0.05, 0.5)
        
        return base_score
    
    def get_dominant_emotions(self, top_k: int = 3) -> List[str]:
        """Get dominant emotions from affect vectors"""
        from ..config import GOEMO_LABEL2IDX
        
        if not hasattr(self, '_emotion_cache'):
            self._emotion_cache = {}
        
        cache_key = f"dominant_{top_k}"
        if cache_key in self._emotion_cache:
            return self._emotion_cache[cache_key]
        
        combined_affect = [0.0] * 28
        
        if self.user_affect_vector:
            combined_affect = [a + b for a, b in zip(combined_affect, self.user_affect_vector)]
        if self.self_affect_vector:
            combined_affect = [a + b for a, b in zip(combined_affect, self.self_affect_vector)]
        if self.primary_affect_vector:
            combined_affect = [a + b for a, b in zip(combined_affect, self.primary_affect_vector)]
        
        # Get top emotions
        emotion_scores = list(enumerate(combined_affect))
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        
        emotions = []
        for i in range(min(top_k, len(emotion_scores))):
            idx, score = emotion_scores[i]
            if score > 0.1:  # Threshold for significance
                emotion_name = [k for k, v in GOEMO_LABEL2IDX.items() if v == idx][0]
                emotions.append(emotion_name)
        
        self._emotion_cache[cache_key] = emotions if emotions else ["neutral"]
        return self._emotion_cache[cache_key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        
        # Convert enums to strings
        data["node_type"] = self.node_type.value
        data["lifecycle_state"] = self.lifecycle_state.value
        data["origin"] = self.origin.value
        
        # Convert timestamps to ISO format
        data["timestamp"] = self.timestamp.isoformat()
        if self.last_echo_timestamp:
            data["last_echo_timestamp"] = self.last_echo_timestamp.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create MemoryNode from dictionary"""
        # Convert enum strings back to enums
        if "node_type" in data:
            data["node_type"] = MemoryNodeType(data["node_type"])
        if "lifecycle_state" in data:
            data["lifecycle_state"] = MemoryLifecycleState(data["lifecycle_state"])
        if "origin" in data:
            data["origin"] = MemoryOrigin(data["origin"])
        
        # Convert timestamp strings back to datetime
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "last_echo_timestamp" in data and data["last_echo_timestamp"]:
            data["last_echo_timestamp"] = datetime.fromisoformat(data["last_echo_timestamp"])
        
        return cls(**data)
    
    def __str__(self) -> str:
        return f"MemoryNode({self.node_id[:8]}:{self.node_type.value}:{self.synopsis[:30]}...)"
    
    def __repr__(self) -> str:
        return (f"MemoryNode(node_id='{self.node_id}', node_type={self.node_type}, "
                f"lifecycle_state={self.lifecycle_state}, emotional_significance={self.emotional_significance:.3f})") 