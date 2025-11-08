import uuid
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone

from .memory_node import MemoryNode, MemoryNodeType, MemoryOrigin, MemoryLifecycleState
from ..config import TEST_MODE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UNIFIED MEMORY STORAGE INTERFACE - PHASE 1.2
# ---------------------------------------------------------------------------

class UnifiedMemoryStorage:
    """
    Unified storage interface for all memory types in the lattice system.
    
    This class provides a single interface for storing memories while maintaining
    complete backwards compatibility with existing storage functions.
    """
    
    def __init__(self):
        self.storage_stats = {
            "total_stored": 0,
            "by_type": {},
            "by_origin": {},
            "failures": 0
        }
        
        # Feature flags for gradual rollout
        self.use_new_storage = True
        self.validate_against_legacy = True
        self.legacy_fallback_enabled = True
    
    async def store_memory_node(self, memory_node: MemoryNode, 
                               semantic_embedding: Optional[List[float]] = None) -> str:
        """
        Store a unified memory node to both ChromaDB and Neo4j.
        
        Args:
            memory_node: The MemoryNode to store
            semantic_embedding: Optional pre-computed semantic embedding
            
        Returns:
            str: The node ID of the stored memory
        """
        try:
            if TEST_MODE:
                logger.info(f"🧪 TEST MODE: Skipping memory storage for {memory_node.node_id[:8]}")
                return memory_node.node_id
            
            # Generate semantic embedding if not provided
            if semantic_embedding is None:
                semantic_embedding = await self._generate_semantic_embedding(memory_node.content)
            
            # Store the embedding in the memory node
            memory_node.semantic_embedding = semantic_embedding
            
            # Store in ChromaDB
            await self._store_to_chroma(memory_node, semantic_embedding)
            
            # Store in Neo4j if available
            try:
                await self._store_to_neo4j(memory_node)
            except RuntimeError as e:
                if "Neo4j not initialized" in str(e):
                    logger.warning(f"⚠️ Neo4j not available for {memory_node.node_id[:8]}, continuing with ChromaDB-only storage")
                else:
                    raise
            
            # Update statistics
            self._update_storage_stats(memory_node)
            
            logger.info(f"✅ Stored unified memory node {memory_node.node_id[:8]} "
                       f"({memory_node.node_type.value}) with emotional significance {memory_node.emotional_significance:.3f}")
            
            return memory_node.node_id
            
        except Exception as e:
            logger.error(f"❌ Error storing memory node {memory_node.node_id[:8]}: {e}")
            self.storage_stats["failures"] += 1
            
            if self.legacy_fallback_enabled:
                logger.info(f"🔄 Attempting legacy fallback for {memory_node.node_id[:8]}")
                return await self._legacy_fallback_store(memory_node)
            
            raise
    
    async def _generate_semantic_embedding(self, content: str) -> List[float]:
        """Generate semantic embedding for content"""
        try:
            from ..config import embedder, EMBED_DIM
            if embedder is None:
                logger.warning("Embedder not initialized, using zero embedding")
                return [0.0] * EMBED_DIM  # Use configured embedding dimension
            
            embedding = embedder.encode([content])[0].tolist()
            
            # Validate embedding dimension
            if len(embedding) != EMBED_DIM:
                logger.error(f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(embedding)}")
                return [0.0] * EMBED_DIM
            
            logger.debug(f"Generated semantic embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating semantic embedding: {e}")
            from ..config import EMBED_DIM
            return [0.0] * EMBED_DIM
    
    async def _store_to_chroma(self, memory_node: MemoryNode, semantic_embedding: List[float]):
        """Store memory node to ChromaDB"""
        from ..config import chroma_db, EMBED_DIM
        if chroma_db is None:
            raise RuntimeError("ChromaDB not initialized")
        
        # Validate semantic embedding dimension
        if len(semantic_embedding) != EMBED_DIM:
            logger.error(f"Invalid semantic embedding dimension: expected {EMBED_DIM}, got {len(semantic_embedding)}")
            semantic_embedding = [0.0] * EMBED_DIM
        
        # Prepare embedding for ChromaDB
        # For dual-affect nodes, combine semantic + user_affect
        # For single-affect nodes, combine semantic + primary_affect
        storage_embedding = semantic_embedding.copy()
        
        if memory_node.node_type == MemoryNodeType.DUAL_AFFECT and memory_node.user_affect_vector:
            if len(memory_node.user_affect_vector) != 28:
                logger.error(f"Invalid user affect vector dimension: expected 28, got {len(memory_node.user_affect_vector)}")
                storage_embedding.extend([0.0] * 28)
            else:
                storage_embedding.extend(memory_node.user_affect_vector)
        elif memory_node.node_type == MemoryNodeType.CUSTOM and memory_node.user_affect_vector:
            # Handle custom nodes like emotional seeds - use user_affect for retrieval
            if len(memory_node.user_affect_vector) != 28:
                logger.error(f"Invalid user affect vector dimension: expected 28, got {len(memory_node.user_affect_vector)}")
                storage_embedding.extend([0.0] * 28)
            else:
                storage_embedding.extend(memory_node.user_affect_vector)
        elif memory_node.primary_affect_vector:
            if len(memory_node.primary_affect_vector) != 28:
                logger.error(f"Invalid primary affect vector dimension: expected 28, got {len(memory_node.primary_affect_vector)}")
                storage_embedding.extend([0.0] * 28)
            else:
                storage_embedding.extend(memory_node.primary_affect_vector)
        else:
            storage_embedding.extend([0.0] * 28)
        
        # Final validation of storage embedding
        expected_total_dim = EMBED_DIM + 28  # semantic + affect
        if len(storage_embedding) != expected_total_dim:
            logger.error(f"Invalid storage embedding dimension: expected {expected_total_dim}, got {len(storage_embedding)}")
            raise RuntimeError(f"Storage embedding dimension mismatch: expected {expected_total_dim}, got {len(storage_embedding)}")
        
        logger.debug(f"Storing memory node {memory_node.node_id[:8]} with embedding dimensions: "
                    f"semantic={len(semantic_embedding)}, affect={len(storage_embedding)-len(semantic_embedding)}, "
                    f"total={len(storage_embedding)}")
        
        # Get metadata for ChromaDB
        metadata = memory_node.to_chroma_metadata()
        
        # Store in ChromaDB
        chroma_db.add(
            documents=[memory_node.content],
            embeddings=[storage_embedding],
            metadatas=[metadata],
            ids=[memory_node.node_id]
        )
    
    async def _store_to_neo4j(self, memory_node: MemoryNode):
        """Store memory node to Neo4j"""
        from ..config import neo4j_conn
        if neo4j_conn is None:
            raise RuntimeError("Neo4j not initialized")
        
        # Get properties and label
        properties = memory_node.to_neo4j_properties()
        label = memory_node.get_neo4j_label()
        
        # Create the node
        with neo4j_conn.session() as session:
            session.run(f"""
                CREATE (n:{label} $properties)
            """, properties=properties)
            
            # Create relationships if specified
            if memory_node.parent_node_id:
                session.run("""
                    MATCH (child {id: $child_id}), (parent {id: $parent_id})
                    CREATE (parent)-[:PARENT_OF]->(child)
                """, child_id=memory_node.node_id, parent_id=memory_node.parent_node_id)
            
            # Create related relationships
            for related_id in memory_node.related_node_ids:
                session.run("""
                    MATCH (node1 {id: $node1_id}), (node2 {id: $node2_id})
                    CREATE (node1)-[:RELATED_TO]->(node2)
                """, node1_id=memory_node.node_id, node2_id=related_id)
    
    async def _legacy_fallback_store(self, memory_node: MemoryNode) -> str:
        """Fallback to legacy storage methods"""
        try:
            if memory_node.node_type == MemoryNodeType.DUAL_AFFECT:
                from .storage import store_dual_affect_node_with_id
                legacy_data = memory_node.to_legacy_dual_affect_format()
                return await store_dual_affect_node_with_id(
                    legacy_data["node_id"],
                    legacy_data["msg"],
                    legacy_data["user_affect"],
                    legacy_data["self_affect"],
                    legacy_data["synopsis"],
                    legacy_data["reflection"],
                    legacy_data["origin"]
                )
            else:
                from .storage import store_smg_node
                legacy_data = memory_node.to_legacy_smg_format()
                return await store_smg_node(
                    legacy_data["msg"],
                    legacy_data["affect_vec"],
                    legacy_data["synopsis"],
                    legacy_data["reflection"],
                    legacy_data["origin"]
                )
        except Exception as e:
            logger.error(f"❌ Legacy fallback failed for {memory_node.node_id[:8]}: {e}")
            raise
    
    def _update_storage_stats(self, memory_node: MemoryNode):
        """Update storage statistics"""
        self.storage_stats["total_stored"] += 1
        
        # Track by type
        node_type = memory_node.node_type.value
        if node_type not in self.storage_stats["by_type"]:
            self.storage_stats["by_type"][node_type] = 0
        self.storage_stats["by_type"][node_type] += 1
        
        # Track by origin
        origin = memory_node.origin.value
        if origin not in self.storage_stats["by_origin"]:
            self.storage_stats["by_origin"][origin] = 0
        self.storage_stats["by_origin"][origin] += 1
    
    async def update_memory_node(self, memory_node: MemoryNode):
        """Update an existing memory node"""
        try:
            if TEST_MODE:
                logger.info(f"🧪 TEST MODE: Skipping memory update for {memory_node.node_id[:8]}")
                return
            
            # Update ChromaDB
            await self._update_chroma_node(memory_node)
            
            # Update Neo4j
            await self._update_neo4j_node(memory_node)
            
            logger.info(f"✅ Updated memory node {memory_node.node_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ Error updating memory node {memory_node.node_id[:8]}: {e}")
            raise
    
    async def _update_chroma_node(self, memory_node: MemoryNode):
        """Update memory node in ChromaDB"""
        from ..config import chroma_db
        if chroma_db is None:
            raise RuntimeError("ChromaDB not initialized")
        
        # Get updated metadata
        metadata = memory_node.to_chroma_metadata()
        
        # Update ChromaDB
        chroma_db.update(
            ids=[memory_node.node_id],
            metadatas=[metadata]
        )
    
    async def _update_neo4j_node(self, memory_node: MemoryNode):
        """Update memory node in Neo4j"""
        from ..config import neo4j_conn
        if neo4j_conn is None:
            raise RuntimeError("Neo4j not initialized")
        
        # Get updated properties
        properties = memory_node.to_neo4j_properties()
        
        # Update Neo4j node
        with neo4j_conn.session() as session:
            session.run("""
                MATCH (n {id: $node_id})
                SET n += $properties
            """, node_id=memory_node.node_id, properties=properties)
    
    async def retrieve_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory node by ID"""
        try:
            from ..config import chroma_db
            if chroma_db is None:
                logger.warning("ChromaDB not initialized")
                return None
            
            # Get from ChromaDB
            results = chroma_db.get(ids=[node_id])
            
            if not results or not results['documents']:
                logger.warning(f"Memory node {node_id[:8]} not found")
                return None
            
            # Create MemoryNode from ChromaDB result
            document = results['documents'][0]
            metadata = results['metadatas'][0]
            
            memory_node = MemoryNode.from_chroma_result(document, metadata)
            
            return memory_node
            
        except Exception as e:
            logger.error(f"❌ Error retrieving memory node {node_id[:8]}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.storage_stats.copy()
    
    def reset_storage_stats(self):
        """Reset storage statistics"""
        self.storage_stats = {
            "total_stored": 0,
            "by_type": {},
            "by_origin": {},
            "failures": 0
        }

# ---------------------------------------------------------------------------
# GLOBAL UNIFIED STORAGE INSTANCE
# ---------------------------------------------------------------------------

# Global instance for use throughout the application
unified_storage = UnifiedMemoryStorage()

# ---------------------------------------------------------------------------
# BACKWARDS COMPATIBILITY WRAPPER FUNCTIONS
# ---------------------------------------------------------------------------

async def store_memory_node(memory_node: MemoryNode, 
                           semantic_embedding: Optional[List[float]] = None) -> str:
    """
    Store a unified memory node (new primary interface).
    
    This is the new primary interface for storing memories in the lattice system.
    """
    return await unified_storage.store_memory_node(memory_node, semantic_embedding)

async def create_and_store_smg_memory(content: str, affect_vec: List[float], synopsis: str,
                                     reflection: Optional[str] = None, 
                                     origin: str = "external_user") -> str:
    """Create and store a single memory generation (SMG) node using the new system"""
    memory_node = MemoryNode(
        node_type=MemoryNodeType.SMG,
        content=content,
        synopsis=synopsis,
        reflection=reflection,
        primary_affect_vector=affect_vec,
        origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.EXTERNAL_USER
    )
    
    return await store_memory_node(memory_node)

async def create_and_store_dual_affect_memory(content: str, user_affect: List[float], 
                                             self_affect: List[float], synopsis: str,
                                             reflection: Optional[str] = None,
                                             origin: str = "dual_channel") -> str:
    """Create and store a dual-affect memory node using the new system"""
    memory_node = MemoryNode(
        node_type=MemoryNodeType.DUAL_AFFECT,
        content=content,
        synopsis=synopsis,
        reflection=reflection,
        user_affect_vector=user_affect,
        self_affect_vector=self_affect,
        origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.DUAL_CHANNEL
    )
    
    return await store_memory_node(memory_node)

async def create_and_store_dual_affect_memory_with_id(node_id: str, content: str, 
                                                     user_affect: List[float], 
                                                     self_affect: List[float], 
                                                     synopsis: str,
                                                     reflection: Optional[str] = None,
                                                     origin: str = "dual_channel") -> str:
    """Create and store a dual-affect memory node with predefined ID"""
    memory_node = MemoryNode(
        node_id=node_id,
        node_type=MemoryNodeType.DUAL_AFFECT,
        content=content,
        synopsis=synopsis,
        reflection=reflection,
        user_affect_vector=user_affect,
        self_affect_vector=self_affect,
        origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.DUAL_CHANNEL
    )
    
    return await store_memory_node(memory_node)

async def update_memory_node_affect(node_id: str, self_affect: List[float]):
    """Update an existing memory node with self-affect data"""
    memory_node = await unified_storage.retrieve_memory_node(node_id)
    if memory_node:
        memory_node.self_affect_vector = self_affect
        memory_node._calculate_emotional_significance()  # Recalculate significance
        await unified_storage.update_memory_node(memory_node)
    else:
        logger.warning(f"Memory node {node_id[:8]} not found for affect update")

async def update_memory_node_reflections(node_id: str, self_affect: List[float], 
                                       user_reflection: str, self_reflection: str):
    """Update memory node with self-affect and both reflections"""
    memory_node = await unified_storage.retrieve_memory_node(node_id)
    if memory_node:
        memory_node.self_affect_vector = self_affect
        memory_node.reflection = user_reflection
        # Store self-reflection in legacy metadata for backwards compatibility
        memory_node.legacy_metadata["self_reflection"] = self_reflection
        memory_node._calculate_emotional_significance()
        await unified_storage.update_memory_node(memory_node)
    else:
        logger.warning(f"Memory node {node_id[:8]} not found for reflection update")

# ---------------------------------------------------------------------------
# LEGACY WRAPPER FUNCTIONS FOR BACKWARDS COMPATIBILITY
# ---------------------------------------------------------------------------

async def store_smg_node_unified(msg: str, affect_vec: List[float], synopsis: str, 
                                reflection: Optional[str] = None, origin: str = "external_user") -> str:
    """
    Unified wrapper for legacy store_smg_node function.
    
    This maintains the exact same interface as the legacy function while using
    the new unified storage system internally.
    """
    return await create_and_store_smg_memory(msg, affect_vec, synopsis, reflection, origin)

async def store_dual_affect_node_unified(msg: str, user_affect: List[float], 
                                       self_affect: List[float], synopsis: str,
                                       reflection: Optional[str] = None,
                                       origin: str = "dual_channel", session_id: Optional[str] = None, 
                                       turn_id: Optional[str] = None) -> str:
    """
    Unified wrapper for legacy store_dual_affect_node function.
    """
    memory_node = MemoryNode(
        node_type=MemoryNodeType.DUAL_AFFECT,
        content=msg,
        synopsis=synopsis,
        reflection=reflection,
        user_affect_vector=user_affect,
        self_affect_vector=self_affect,
        origin=MemoryOrigin(origin) if origin in [o.value for o in MemoryOrigin] else MemoryOrigin.DUAL_CHANNEL
    )
    
    # Add session and turn information to legacy metadata
    if session_id:
        memory_node.legacy_metadata["session_id"] = session_id
    if turn_id:
        memory_node.legacy_metadata["turn_id"] = turn_id
    
    return await store_memory_node(memory_node)

async def store_dual_affect_node_with_id_unified(node_id: str, msg: str, 
                                               user_affect: List[float], 
                                               self_affect: List[float], 
                                               synopsis: str,
                                               reflection: Optional[str] = None,
                                               origin: str = "dual_channel") -> str:
    """
    Unified wrapper for legacy store_dual_affect_node_with_id function.
    """
    return await create_and_store_dual_affect_memory_with_id(node_id, msg, user_affect, self_affect, synopsis, reflection, origin)

# ---------------------------------------------------------------------------
# STORAGE VALIDATION AND DIAGNOSTICS
# ---------------------------------------------------------------------------

async def validate_unified_storage():
    """Validate that the unified storage system is working correctly"""
    try:
        # Test storage of different memory types
        test_results = {
            "smg_test": False,
            "dual_affect_test": False,
            "retrieval_test": False,
            "update_test": False
        }
        
        logger.info("🔍 Validating unified storage system...")
        
        # Test SMG storage
        try:
            test_smg = MemoryNode(
                content="Test SMG memory",
                synopsis="Test synopsis",
                primary_affect_vector=[0.1] * 28,
                node_type=MemoryNodeType.SMG
            )
            node_id = await store_memory_node(test_smg)
            test_results["smg_test"] = bool(node_id)
            logger.info("✅ SMG storage test passed")
        except Exception as e:
            logger.error(f"❌ SMG storage test failed: {e}")
        
        # Test dual-affect storage
        try:
            test_dual = MemoryNode(
                content="Test dual-affect memory",
                synopsis="Test dual synopsis",
                user_affect_vector=[0.2] * 28,
                self_affect_vector=[0.3] * 28,
                node_type=MemoryNodeType.DUAL_AFFECT
            )
            dual_id = await store_memory_node(test_dual)
            test_results["dual_affect_test"] = bool(dual_id)
            logger.info("✅ Dual-affect storage test passed")
        except Exception as e:
            logger.error(f"❌ Dual-affect storage test failed: {e}")
        
        # Test retrieval
        try:
            if test_results["smg_test"]:
                retrieved = await unified_storage.retrieve_memory_node(node_id)
                test_results["retrieval_test"] = retrieved is not None
                logger.info("✅ Retrieval test passed")
        except Exception as e:
            logger.error(f"❌ Retrieval test failed: {e}")
        
        # Test update
        try:
            if test_results["retrieval_test"]:
                retrieved.echo_count += 1
                await unified_storage.update_memory_node(retrieved)
                test_results["update_test"] = True
                logger.info("✅ Update test passed")
        except Exception as e:
            logger.error(f"❌ Update test failed: {e}")
        
        all_passed = all(test_results.values())
        logger.info(f"🎯 Unified storage validation {'PASSED' if all_passed else 'FAILED'}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Storage validation error: {e}")
        return {"error": str(e)}

async def get_unified_storage_diagnostics():
    """Get comprehensive diagnostics for the unified storage system"""
    diagnostics = {
        "storage_stats": unified_storage.get_storage_stats(),
        "feature_flags": {
            "use_new_storage": unified_storage.use_new_storage,
            "validate_against_legacy": unified_storage.validate_against_legacy,
            "legacy_fallback_enabled": unified_storage.legacy_fallback_enabled
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return diagnostics 