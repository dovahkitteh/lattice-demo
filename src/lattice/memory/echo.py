import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from ..config import TEST_MODE
from .memory_node import MemoryNode, MemoryLifecycleState
from .memory_node import MemoryNodeType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENHANCED ECHO FUNCTIONS - PHASE 1.4
# ---------------------------------------------------------------------------

async def echo_update_unified(memory_node: MemoryNode, query_text: str, query_affect: List[float]):
    """
    Enhanced echo update that works with unified MemoryNode architecture.
    
    This function updates memory access patterns and lifecycle states based on
    interaction patterns, providing the foundation for intelligent memory evolution.
    """
    try:
        if TEST_MODE:
            logger.info(f"🧪 TEST MODE: Skipping echo update for {memory_node.node_id[:8]}")
            return
        
        # Update the memory node's echo information
        memory_node.update_echo_access(query_text, query_affect)
        
        # ENHANCED: Special handling for emotional memory seeds
        if memory_node.node_type == MemoryNodeType.CUSTOM:
            try:
                from .emotional_seed_enhancement import emotional_seed_enhancement
                
                # Validate memory_node before processing
                if not isinstance(memory_node, MemoryNode):
                    logger.warning("Invalid memory_node for emotional seed echo processing")
                else:
                    # Integrate with echo system
                    await emotional_seed_enhancement.integrate_with_echo_system(memory_node)
                    # Integrate with lifecycle system
                    await emotional_seed_enhancement.integrate_with_lifecycle_system(memory_node)
                    
                    logger.debug(f"🎭 Enhanced echo processing for emotional seed {memory_node.node_id[:8]}")
                
            except Exception as e:
                logger.warning(f"Error in emotional seed echo integration: {e}")
                # Continue with regular echo processing
        
        # Update storage with new echo information
        from .unified_storage import unified_storage
        await unified_storage.update_memory_node(memory_node)
        
        # Create echo relationships for strong echoes
        if memory_node.echo_strength > 5.0:  # Threshold for strong echo
            await create_unified_echo_relationship(memory_node, query_text, query_affect)
        
        # Analyze access patterns for lifecycle transitions
        await analyze_memory_lifecycle_transition(memory_node)
        
        logger.debug(f"🔄 Enhanced echo update for {memory_node.node_id[:8]} "
                    f"(echoes: {memory_node.echo_count}, strength: {memory_node.echo_strength:.3f}, "
                    f"state: {memory_node.lifecycle_state.value})")
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced echo update for {memory_node.node_id[:8]}: {e}")

async def echo_update(node_id: str, affect_vec: list[float], query_text: str):
    """
    Legacy echo update function - enhanced to work with unified storage when available.
    
    This maintains backward compatibility while providing enhanced capabilities
    when the unified storage system is enabled.
    """
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Skipping echo update")
            return
        
        # Try to get the memory node from unified storage
        from .unified_storage import unified_storage
        from . import ENABLE_UNIFIED_STORAGE
        
        if ENABLE_UNIFIED_STORAGE:
            memory_node = await unified_storage.retrieve_memory_node(node_id)
            if memory_node:
                # Use enhanced echo update
                await echo_update_unified(memory_node, query_text, affect_vec)
                return
        
        # Fallback to legacy echo update
        await echo_update_legacy(node_id, affect_vec, query_text)
        
    except Exception as e:
        logger.error(f"❌ Error in echo update for node {node_id[:8]}: {e}")

async def echo_update_legacy(node_id: str, affect_vec: list[float], query_text: str):
    """Legacy echo update implementation for backward compatibility"""
    try:
        from ..config import chroma_db, neo4j_conn
        if not chroma_db or not neo4j_conn:
            logger.warning("Database not initialized for echo update")
            return
        
        # Update ChromaDB metadata to track echo
        existing = chroma_db.get(ids=[node_id])
        if existing and existing['metadatas']:
            metadata = existing['metadatas'][0]
            echo_count = metadata.get('echo_count', 0) + 1
            metadata['echo_count'] = echo_count
            metadata['last_echo'] = datetime.now(timezone.utc).isoformat()
            metadata['last_echo_query'] = query_text[:100]  # Truncate long queries
            
            chroma_db.update(
                ids=[node_id],
                metadatas=[metadata]
            )
        
        # Update Neo4j with echo information
        with neo4j_conn.session() as session:
            session.run("""
                MATCH (n {id: $node_id})
                SET n.echo_count = COALESCE(n.echo_count, 0) + 1,
                    n.last_echo = $timestamp,
                    n.last_echo_query = $query,
                    n.echo_strength = COALESCE(n.echo_strength, 0.0) + $affect_magnitude
            """, {
                "node_id": node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query_text[:100],
                "affect_magnitude": sum(abs(x) for x in affect_vec)
            })
        
        # Create echo relationship if this is a strong echo (high affect)
        affect_magnitude = sum(abs(x) for x in affect_vec)
        if affect_magnitude > 2.0:  # Threshold for strong echo
            await create_echo_relationship(node_id, query_text, affect_magnitude)
        
        logger.debug(f"🔄 Legacy echo update for node {node_id[:8]} (affect: {affect_magnitude:.3f})")
    
    except Exception as e:
        logger.error(f"❌ Error in legacy echo update for node {node_id[:8]}: {e}")

async def create_unified_echo_relationship(memory_node: MemoryNode, query_text: str, query_affect: List[float]):
    """Create enhanced echo relationship with unified memory architecture"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return
        
        # Create a comprehensive echo node with enhanced metadata
        echo_id = f"echo_{memory_node.node_id}_{int(datetime.now().timestamp())}"
        query_magnitude = sum(abs(x) for x in query_affect) if query_affect else 0.0
        
        with neo4j_conn.session() as session:
            session.run("""
                MATCH (n {id: $node_id})
                CREATE (e:EnhancedEchoNode {
                    id: $echo_id,
                    source_node: $node_id,
                    query_text: $query_text,
                    query_affect_magnitude: $query_magnitude,
                    memory_emotional_significance: $memory_emotional_significance,
                    memory_lifecycle_state: $memory_lifecycle_state,
                    memory_node_type: $memory_node_type,
                    echo_strength_at_time: $echo_strength,
                    echo_count_at_time: $echo_count,
                    timestamp: $timestamp
                })
                CREATE (n)-[:ENHANCED_ECHO]->(e)
            """, {
                "node_id": memory_node.node_id,
                "echo_id": echo_id,
                "query_text": query_text,
                "query_magnitude": query_magnitude,
                "memory_emotional_significance": memory_node.emotional_significance,
                "memory_lifecycle_state": memory_node.lifecycle_state.value,
                "memory_node_type": memory_node.node_type.value,
                "echo_strength": memory_node.echo_strength,
                "echo_count": memory_node.echo_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(f"🔊 Created enhanced echo relationship for {memory_node.node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error creating enhanced echo relationship: {e}")

async def analyze_memory_lifecycle_transition(memory_node: MemoryNode):
    """Analyze memory access patterns to determine lifecycle transitions"""
    try:
        previous_state = memory_node.lifecycle_state
        
        # Implement lifecycle transition logic
        if memory_node.lifecycle_state == MemoryLifecycleState.RAW:
            # Transition to ECHOED after first few accesses
            if memory_node.echo_count >= 2:
                memory_node.lifecycle_state = MemoryLifecycleState.ECHOED
                logger.info(f"🔄 Memory {memory_node.node_id[:8]} transitioned RAW → ECHOED")
        
        elif memory_node.lifecycle_state == MemoryLifecycleState.ECHOED:
            # Transition to CRYSTALLIZED based on echo strength and emotional significance
            crystallization_threshold = 10.0
            significance_bonus = memory_node.emotional_significance * 0.5
            
            if memory_node.echo_strength + significance_bonus > crystallization_threshold:
                memory_node.lifecycle_state = MemoryLifecycleState.CRYSTALLIZED
                logger.info(f"🔄 Memory {memory_node.node_id[:8]} transitioned ECHOED → CRYSTALLIZED")
        
        # Update importance score based on lifecycle state
        await update_memory_importance(memory_node)
        
        # Log lifecycle transition if it occurred
        if previous_state != memory_node.lifecycle_state:
            await log_lifecycle_transition(memory_node, previous_state, memory_node.lifecycle_state)
        
    except Exception as e:
        logger.error(f"❌ Error analyzing lifecycle transition for {memory_node.node_id[:8]}: {e}")

async def update_memory_importance(memory_node: MemoryNode):
    """Update memory importance score based on access patterns and lifecycle state"""
    try:
        # Base importance from emotional significance
        base_importance = memory_node.emotional_significance * 0.1
        
        # Echo frequency bonus
        echo_bonus = min(memory_node.echo_count * 0.05, 0.5)
        
        # Echo strength bonus
        strength_bonus = min(memory_node.echo_strength * 0.02, 0.3)
        
        # Lifecycle state bonus
        lifecycle_bonuses = {
            MemoryLifecycleState.RAW: 0.0,
            MemoryLifecycleState.ECHOED: 0.1,
            MemoryLifecycleState.CRYSTALLIZED: 0.3,
            MemoryLifecycleState.ARCHIVED: -0.1,
            MemoryLifecycleState.COMPRESSED: -0.2,
            MemoryLifecycleState.SHADOW: 0.2,
        }
        lifecycle_bonus = lifecycle_bonuses.get(memory_node.lifecycle_state, 0.0)
        
        # Calculate new importance score
        new_importance = base_importance + echo_bonus + strength_bonus + lifecycle_bonus
        
        # Update if significantly changed
        if abs(new_importance - memory_node.importance_score) > 0.05:
            memory_node.importance_score = new_importance
            logger.debug(f"📊 Updated importance for {memory_node.node_id[:8]}: {new_importance:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error updating memory importance for {memory_node.node_id[:8]}: {e}")

async def log_lifecycle_transition(memory_node: MemoryNode, old_state: MemoryLifecycleState, new_state: MemoryLifecycleState):
    """Log memory lifecycle transitions for analysis"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return
        
        # Create lifecycle transition log
        transition_id = f"transition_{memory_node.node_id}_{int(datetime.now().timestamp())}"
        
        with neo4j_conn.session() as session:
            session.run("""
                MATCH (n {id: $node_id})
                CREATE (t:LifecycleTransition {
                    id: $transition_id,
                    source_node: $node_id,
                    old_state: $old_state,
                    new_state: $new_state,
                    echo_count_at_transition: $echo_count,
                    echo_strength_at_transition: $echo_strength,
                    emotional_significance: $emotional_significance,
                    importance_score: $importance_score,
                    timestamp: $timestamp
                })
                CREATE (n)-[:LIFECYCLE_TRANSITION]->(t)
            """, {
                "node_id": memory_node.node_id,
                "transition_id": transition_id,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "echo_count": memory_node.echo_count,
                "echo_strength": memory_node.echo_strength,
                "emotional_significance": memory_node.emotional_significance,
                "importance_score": memory_node.importance_score,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(f"📝 Logged lifecycle transition for {memory_node.node_id[:8]}: {old_state.value} → {new_state.value}")
    
    except Exception as e:
        logger.error(f"❌ Error logging lifecycle transition: {e}")

async def create_echo_relationship(node_id: str, query_text: str, affect_magnitude: float):
    """Create an echo relationship in Neo4j for strong echoes (legacy compatibility)"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return
        
        # Create a virtual echo node and relationship
        echo_id = f"echo_{node_id}_{int(datetime.now().timestamp())}"
        
        with neo4j_conn.session() as session:
            session.run("""
                MATCH (n {id: $node_id})
                CREATE (e:EchoNode {
                    id: $echo_id,
                    source_node: $node_id,
                    query_text: $query_text,
                    affect_magnitude: $affect_magnitude,
                    timestamp: $timestamp
                })
                CREATE (n)-[:ECHOED_BY]->(e)
            """, {
                "node_id": node_id,
                "echo_id": echo_id,
                "query_text": query_text,
                "affect_magnitude": affect_magnitude,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(f"🔊 Created strong echo relationship for node {node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error creating echo relationship: {e}")

async def get_enhanced_echo_statistics():
    """Get enhanced statistics about echo patterns and memory lifecycle"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return {"error": "Neo4j not initialized"}
        
        with neo4j_conn.session() as session:
            # Get lifecycle state distribution
            result = session.run("""
                MATCH (n)
                WHERE n.lifecycle_state IS NOT NULL
                RETURN n.lifecycle_state as state, count(*) as count
                ORDER BY count DESC
            """)
            
            lifecycle_distribution = {}
            for record in result:
                lifecycle_distribution[record["state"]] = record["count"]
            
            # Get top crystallized memories
            result = session.run("""
                MATCH (n)
                WHERE n.lifecycle_state = 'crystallized'
                RETURN n.id as node_id, n.echo_count as echo_count, 
                       n.echo_strength as echo_strength, n.emotional_significance as emotional_significance,
                       n.importance_score as importance_score
                ORDER BY n.importance_score DESC
                LIMIT 10
            """)
            
            top_crystallized = []
            for record in result:
                top_crystallized.append({
                    "node_id": record["node_id"],
                    "echo_count": record["echo_count"],
                    "echo_strength": record["echo_strength"],
                    "emotional_significance": record["emotional_significance"],
                    "importance_score": record["importance_score"]
                })
            
            # Get recent lifecycle transitions
            result = session.run("""
                MATCH (t:LifecycleTransition)
                RETURN t.id as transition_id, t.source_node as source_node,
                       t.old_state as old_state, t.new_state as new_state,
                       t.timestamp as timestamp
                ORDER BY t.timestamp DESC
                LIMIT 20
            """)
            
            recent_transitions = []
            for record in result:
                recent_transitions.append({
                    "transition_id": record["transition_id"],
                    "source_node": record["source_node"],
                    "old_state": record["old_state"],
                    "new_state": record["new_state"],
                    "timestamp": record["timestamp"]
                })
            
            # Get enhanced echo activity
            result = session.run("""
                MATCH (e:EnhancedEchoNode)
                RETURN e.id as echo_id, e.source_node as source_node,
                       e.query_affect_magnitude as query_magnitude,
                       e.memory_lifecycle_state as memory_state,
                       e.timestamp as timestamp
                ORDER BY e.timestamp DESC
                LIMIT 20
            """)
            
            enhanced_echoes = []
            for record in result:
                enhanced_echoes.append({
                    "echo_id": record["echo_id"],
                    "source_node": record["source_node"],
                    "query_magnitude": record["query_magnitude"],
                    "memory_state": record["memory_state"],
                    "timestamp": record["timestamp"]
                })
            
            return {
                "lifecycle_distribution": lifecycle_distribution,
                "top_crystallized_memories": top_crystallized,
                "recent_lifecycle_transitions": recent_transitions,
                "enhanced_echo_activity": enhanced_echoes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error getting enhanced echo statistics: {e}")
        return {"error": str(e)}

async def get_echo_statistics():
    """Get statistics about echo patterns (legacy compatibility)"""
    try:
        from . import ENABLE_UNIFIED_STORAGE
        
        if ENABLE_UNIFIED_STORAGE:
            # Return enhanced statistics when unified storage is enabled
            return await get_enhanced_echo_statistics()
        
        # Fallback to legacy statistics
        from ..config import neo4j_conn
        if not neo4j_conn:
            return {"error": "Neo4j not initialized"}
        
        with neo4j_conn.session() as session:
            # Get nodes with most echoes
            result = session.run("""
                MATCH (n)
                WHERE n.echo_count IS NOT NULL AND n.echo_count > 0
                RETURN n.id as node_id, n.echo_count as echo_count, n.echo_strength as echo_strength
                ORDER BY n.echo_count DESC
                LIMIT 10
            """)
            
            top_echoed = []
            for record in result:
                top_echoed.append({
                    "node_id": record["node_id"],
                    "echo_count": record["echo_count"],
                    "echo_strength": record["echo_strength"]
                })
            
            # Get recent echo activity
            result = session.run("""
                MATCH (e:EchoNode)
                RETURN e.id as echo_id, e.source_node as source_node, 
                       e.affect_magnitude as affect_magnitude, e.timestamp as timestamp
                ORDER BY e.timestamp DESC
                LIMIT 20
            """)
            
            recent_echoes = []
            for record in result:
                recent_echoes.append({
                    "echo_id": record["echo_id"],
                    "source_node": record["source_node"],
                    "affect_magnitude": record["affect_magnitude"],
                    "timestamp": record["timestamp"]
                })
            
            return {
                "top_echoed_nodes": top_echoed,
                "recent_echo_activity": recent_echoes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error getting echo statistics: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# MEMORY ACCESS PATTERN ANALYSIS
# ---------------------------------------------------------------------------

async def analyze_memory_access_patterns():
    """Analyze memory access patterns to identify insights about memory usage"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return {"error": "Neo4j not initialized"}
        
        with neo4j_conn.session() as session:
            # Analyze echo frequency patterns
            result = session.run("""
                MATCH (n)
                WHERE n.echo_count IS NOT NULL
                RETURN 
                    avg(n.echo_count) as avg_echo_count,
                    max(n.echo_count) as max_echo_count,
                    count(n) as total_nodes,
                    count(CASE WHEN n.echo_count > 0 THEN 1 END) as nodes_with_echoes,
                    count(CASE WHEN n.echo_count > 5 THEN 1 END) as highly_echoed_nodes
            """)
            
            patterns = {}
            for record in result:
                patterns = {
                    "average_echo_count": record["avg_echo_count"],
                    "max_echo_count": record["max_echo_count"],
                    "total_nodes": record["total_nodes"],
                    "nodes_with_echoes": record["nodes_with_echoes"],
                    "highly_echoed_nodes": record["highly_echoed_nodes"]
                }
            
            # Calculate engagement metrics
            if patterns["total_nodes"] > 0:
                patterns["echo_engagement_rate"] = patterns["nodes_with_echoes"] / patterns["total_nodes"]
                patterns["high_engagement_rate"] = patterns["highly_echoed_nodes"] / patterns["total_nodes"]
            
            # Analyze lifecycle progression
            result = session.run("""
                MATCH (n)
                WHERE n.lifecycle_state IS NOT NULL
                RETURN n.lifecycle_state as state, 
                       avg(n.echo_count) as avg_echoes,
                       avg(n.emotional_significance) as avg_emotional_significance,
                       count(*) as count
            """)
            
            lifecycle_analysis = {}
            for record in result:
                lifecycle_analysis[record["state"]] = {
                    "count": record["count"],
                    "avg_echoes": record["avg_echoes"],
                    "avg_emotional_significance": record["avg_emotional_significance"]
                }
            
            return {
                "access_patterns": patterns,
                "lifecycle_analysis": lifecycle_analysis,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error analyzing memory access patterns: {e}")
        return {"error": str(e)} 