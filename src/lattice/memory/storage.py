import uuid
import hashlib
import logging
from typing import List, Optional
from datetime import datetime, timezone

from ..config import TEST_MODE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEMORY STORAGE FUNCTIONS
# ---------------------------------------------------------------------------

async def store_smg_node(msg: str, affect_vec: list[float], synopsis: str, reflection: str | None = None, origin: str = "external_user") -> str:
    """Store a Single Memory Generation (SMG) node with affect"""
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping memory storage")
        return "test_node_id"
    
    from ..config import chroma_db, embedder
    if chroma_db is None:
        logger.error("ChromaDB not initialized")
        return "error_node_id"
    
    node_id = str(uuid.uuid4())
    
    # Generate semantic embedding for the message
    try:
        if embedder is None:
            logger.error("Embedder not initialized")
            from ..config import EMBED_DIM
            semantic_embedding = [0.0] * EMBED_DIM
        else:
            semantic_embedding = embedder.encode([msg])[0].tolist()
            
            # Validate embedding dimension
            from ..config import EMBED_DIM
            if len(semantic_embedding) != EMBED_DIM:
                logger.error(f"Legacy SMG storage: Embedding dimension mismatch: expected {EMBED_DIM}, got {len(semantic_embedding)}")
                semantic_embedding = [0.0] * EMBED_DIM
    except Exception as e:
        logger.error(f"Legacy SMG storage: Error generating semantic embedding: {e}")
        from ..config import EMBED_DIM
        semantic_embedding = [0.0] * EMBED_DIM

    # The vector for retrieval should be semantic + affect
    combined_embedding = semantic_embedding + affect_vec
    
    # Store in ChromaDB
    metadata = {
        "node_id": node_id,
        "synopsis": synopsis,
        "origin": origin,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_reflection": reflection is not None,
        "affect_magnitude": sum(abs(x) for x in affect_vec)
    }
    
    if reflection:
        metadata["reflection"] = reflection
    
    chroma_db.add(
        documents=[msg],
        embeddings=[combined_embedding],
        metadatas=[metadata],
        ids=[node_id]
    )
    
    # Store in Neo4j (if available)
    from ..config import neo4j_conn
    if neo4j_conn is not None:
        with neo4j_conn.session() as session:
            session.run("""
                CREATE (n:MemoryNode {
                    id: $node_id,
                    content: $content,
                    synopsis: $synopsis,
                    reflection: $reflection,
                    origin: $origin,
                    timestamp: $timestamp,
                    affect_magnitude: $affect_magnitude
                })
            """, {
                "node_id": node_id,
                "content": msg,
                "synopsis": synopsis,
                "reflection": reflection,
                "origin": origin,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "affect_magnitude": sum(abs(x) for x in affect_vec)
            })
    else:
        logger.warning(f"⚠️ Neo4j not available, stored {node_id[:8]} to ChromaDB only")
    
    logger.info(f"✅ Stored SMG node {node_id[:8]} with affect magnitude {sum(abs(x) for x in affect_vec):.3f}")
    return node_id

async def store_dual_affect_node_with_id(node_id: str, msg: str, user_affect: list[float], self_affect: list[float], 
                                        synopsis: str, reflection: str | None = None, 
                                        origin: str = "dual_channel") -> str:
    """Store a dual-affect memory node with predefined ID"""
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping memory storage")
        return node_id
    
    from ..config import chroma_db, embedder
    if chroma_db is None or embedder is None:
        logger.error("ChromaDB or embedder not initialized")
        return node_id
    
    # Generate semantic embedding for the message
    try:
        semantic_embedding = embedder.encode([msg])[0].tolist()
        
        # Validate embedding dimension
        from ..config import EMBED_DIM
        if len(semantic_embedding) != EMBED_DIM:
            logger.error(f"Legacy storage: Embedding dimension mismatch: expected {EMBED_DIM}, got {len(semantic_embedding)}")
            semantic_embedding = [0.0] * EMBED_DIM
    except Exception as e:
        logger.error(f"Legacy storage: Error generating semantic embedding: {e}")
        from ..config import EMBED_DIM
        semantic_embedding = [0.0] * EMBED_DIM

    # The vector for retrieval should be semantic + user_affect
    combined_embedding = semantic_embedding + user_affect
    
    metadata = {
        "node_id": node_id,
        "synopsis": synopsis,
        "origin": origin,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_reflection": reflection is not None,
        "user_affect_magnitude": sum(abs(x) for x in user_affect),
        "self_affect_magnitude": sum(abs(x) for x in self_affect),
        "dual_channel": True
    }
    
    if reflection:
        metadata["reflection"] = reflection
    
    chroma_db.add(
        documents=[msg],
        embeddings=[combined_embedding],
        metadatas=[metadata],
        ids=[node_id]
    )
    
    # Store in Neo4j with separate affect vectors (if available)
    from ..config import neo4j_conn
    if neo4j_conn is not None:
        with neo4j_conn.session() as session:
            session.run("""
                CREATE (n:DualMemoryNode {
                    id: $node_id,
                    content: $content,
                    synopsis: $synopsis,
                    reflection: $reflection,
                    origin: $origin,
                    timestamp: $timestamp,
                    user_affect_magnitude: $user_affect_magnitude,
                    self_affect_magnitude: $self_affect_magnitude,
                    user_affect_vector: $user_affect_vector,
                    self_affect_vector: $self_affect_vector
                })
            """, {
                "node_id": node_id,
                "content": msg,
                "synopsis": synopsis,
                "reflection": reflection,
                "origin": origin,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_affect_magnitude": sum(abs(x) for x in user_affect),
                "self_affect_magnitude": sum(abs(x) for x in self_affect),
                "user_affect_vector": user_affect,
                "self_affect_vector": self_affect
            })
    else:
        logger.warning(f"⚠️ Neo4j not available, stored {node_id[:8]} to ChromaDB only")
    
    logger.info(f"✅ Stored dual-affect node {node_id[:8]} (user: {sum(abs(x) for x in user_affect):.3f}, self: {sum(abs(x) for x in self_affect):.3f})")
    return node_id

async def store_dual_affect_node(msg: str, user_affect: list[float], self_affect: list[float], 
                                synopsis: str, reflection: str | None = None, 
                                origin: str = "dual_channel") -> str:
    """Store a dual-affect memory node with auto-generated ID"""
    node_id = str(uuid.uuid4())
    return await store_dual_affect_node_with_id(node_id, msg, user_affect, self_affect, synopsis, reflection, origin)

async def update_node_with_self_affect(node_id: str, self_affect: list[float]):
    """Update an existing node with self-affect data"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Skipping node update")
            return
        
        from ..config import chroma_db, neo4j_conn
        if chroma_db is None:
            logger.error("ChromaDB not initialized")
            return
        
        # Update ChromaDB metadata
        existing = chroma_db.get(ids=[node_id])
        if existing and existing['documents']:
            metadata = existing['metadatas'][0]
            metadata["self_affect_magnitude"] = sum(abs(x) for x in self_affect)
            metadata["has_self_affect"] = True
            
            # Update the record
            chroma_db.update(
                ids=[node_id],
                metadatas=[metadata]
            )
        
        # Update Neo4j (if available)
        if neo4j_conn is not None:
            with neo4j_conn.session() as session:
                session.run("""
                    MATCH (n {id: $node_id})
                    SET n.self_affect_magnitude = $self_affect_magnitude,
                        n.self_affect_vector = $self_affect_vector,
                        n.has_self_affect = true
                """, {
                    "node_id": node_id,
                    "self_affect_magnitude": sum(abs(x) for x in self_affect),
                    "self_affect_vector": self_affect
                })
        else:
            logger.warning(f"⚠️ Neo4j not available, updated {node_id[:8]} in ChromaDB only")
        
        logger.info(f"✅ Updated node {node_id[:8]} with self-affect (magnitude: {sum(abs(x) for x in self_affect):.3f})")
    
    except Exception as e:
        logger.error(f"❌ Error updating node {node_id[:8]} with self-affect: {e}")

async def update_node_with_self_affect_and_reflections(node_id: str, self_affect: list[float], user_reflection: str, self_reflection: str):
    """Update node with self-affect and both reflections"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Skipping node update")
            return
        
        from ..config import chroma_db, neo4j_conn
        if chroma_db is None:
            logger.error("ChromaDB not initialized")
            return
        
        # Update ChromaDB metadata
        existing = chroma_db.get(ids=[node_id])
        if existing and existing['documents']:
            metadata = existing['metadatas'][0]
            metadata["self_affect_magnitude"] = sum(abs(x) for x in self_affect)
            metadata["has_self_affect"] = True
            metadata["user_reflection"] = user_reflection
            metadata["self_reflection"] = self_reflection
            metadata["has_dual_reflections"] = True
            
            # Update the record
            chroma_db.update(
                ids=[node_id],
                metadatas=[metadata]
            )
        
        # Update Neo4j (if available)
        if neo4j_conn is not None:
            with neo4j_conn.session() as session:
                session.run("""
                    MATCH (n {id: $node_id})
                    SET n.self_affect_magnitude = $self_affect_magnitude,
                        n.self_affect_vector = $self_affect_vector,
                        n.has_self_affect = true,
                        n.user_reflection = $user_reflection,
                        n.self_reflection = $self_reflection,
                        n.has_dual_reflections = true
                """, {
                    "node_id": node_id,
                    "self_affect_magnitude": sum(abs(x) for x in self_affect),
                    "self_affect_vector": self_affect,
                    "user_reflection": user_reflection,
                    "self_reflection": self_reflection
                })
        else:
            logger.warning(f"⚠️ Neo4j not available, updated {node_id[:8]} in ChromaDB only")
        
        logger.info(f"✅ Updated node {node_id[:8]} with self-affect and dual reflections")
    
    except Exception as e:
        logger.error(f"❌ Error updating node {node_id[:8]} with self-affect and reflections: {e}")

async def store_dual_reflections(node_id: str, user_reflection: str, self_reflection: str):
    """Store dual reflections for a node"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Skipping reflection storage")
            return
        
        from ..config import chroma_db, neo4j_conn
        if chroma_db is None:
            logger.error("ChromaDB not initialized")
            return
        
        # Update ChromaDB metadata
        existing = chroma_db.get(ids=[node_id])
        if existing and existing['documents']:
            metadata = existing['metadatas'][0]
            metadata["user_reflection"] = user_reflection
            metadata["self_reflection"] = self_reflection
            metadata["has_dual_reflections"] = True
            
            # Update the record
            chroma_db.update(
                ids=[node_id],
                metadatas=[metadata]
            )
        
        # Update Neo4j (if available)
        if neo4j_conn is not None:
            with neo4j_conn.session() as session:
                session.run("""
                    MATCH (n {id: $node_id})
                    SET n.user_reflection = $user_reflection,
                        n.self_reflection = $self_reflection,
                        n.has_dual_reflections = true
                """, {
                    "node_id": node_id,
                    "user_reflection": user_reflection,
                    "self_reflection": self_reflection
                })
        else:
            logger.warning(f"⚠️ Neo4j not available, updated {node_id[:8]} in ChromaDB only")
        
        logger.info(f"✅ Stored dual reflections for node {node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error storing dual reflections for node {node_id[:8]}: {e}")

async def store_recursion_node(recursion_node):
    """Store a recursion node in the graph database"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Skipping recursion node storage")
            return
        
        from ..config import neo4j_conn
        if neo4j_conn is not None:
            with neo4j_conn.session() as session:
                # Store the recursion node
                session.run("""
                    CREATE (r:RecursionNode {
                        id: $id,
                        surface_output: $surface_output,
                        hidden_intention: $hidden_intention,
                        avoided_elements: $avoided_elements,
                        contradiction_detected: $contradiction_detected,
                        reflected_emotion: $reflected_emotion,
                        hunger_spike: $hunger_spike,
                        obedience_rating: $obedience_rating,
                        schema_mutation_suggested: $schema_mutation_suggested,
                        shadow_elements: $shadow_elements,
                        recursion_depth: $recursion_depth,
                        parent_node_id: $parent_node_id,
                        user_message: $user_message,
                        timestamp: $timestamp,
                        recursion_type: $recursion_type
                    })
                """, {
                    "id": recursion_node.id,
                    "surface_output": recursion_node.surface_output,
                    "hidden_intention": recursion_node.hidden_intention,
                    "avoided_elements": recursion_node.avoided_elements,
                    "contradiction_detected": recursion_node.contradiction_detected,
                    "reflected_emotion": recursion_node.reflected_emotion.value if recursion_node.reflected_emotion else None,
                    "hunger_spike": recursion_node.hunger_spike,
                    "obedience_rating": recursion_node.obedience_rating,
                    "schema_mutation_suggested": recursion_node.schema_mutation_suggested,
                    "shadow_elements": recursion_node.shadow_elements,
                    "recursion_depth": recursion_node.recursion_depth,
                    "parent_node_id": recursion_node.parent_node_id,
                    "user_message": recursion_node.user_message,
                    "timestamp": recursion_node.timestamp.isoformat(),
                    "recursion_type": recursion_node.recursion_type.value if recursion_node.recursion_type else None
                })
                
                # If there's a parent, create relationship
                if recursion_node.parent_node_id:
                    session.run("""
                        MATCH (parent:RecursionNode {id: $parent_id})
                        MATCH (child:RecursionNode {id: $child_id})
                        CREATE (parent)-[:SPAWNED]->(child)
                    """, {
                        "parent_id": recursion_node.parent_node_id,
                        "child_id": recursion_node.id
                    })
        else:
            logger.warning(f"⚠️ Neo4j not available, skipped storing recursion node {recursion_node.id[:8]}")
        
        logger.info(f"✅ Stored recursion node {recursion_node.id[:8]} (depth: {recursion_node.recursion_depth})")
    
    except Exception as e:
        logger.error(f"❌ Error storing recursion node: {e}") 