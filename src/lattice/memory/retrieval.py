import logging
import asyncio
from typing import List
from datetime import datetime, timezone

from ..config import EMBED_DIM
from ..models import Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEMORY RETRIEVAL FUNCTIONS
# ---------------------------------------------------------------------------

async def retrieve_context(query: str, q_affect: list[float], k: int = 10) -> list[str]:
    """Retrieve relevant context from memory using dual semantic + affective search"""
    try:
        from ..config import chroma_db, embedder
        if not chroma_db or not embedder:
            logger.warning("Database or embedder not initialized, returning empty context")
            return []
        
        # Generate semantic embedding for the query
        query_embedding = embedder.encode([query])[0].tolist()
        
        # Combine semantic and affective vectors
        combined_query_vector = query_embedding + q_affect
        
        # Query ChromaDB with the combined vector
        results = chroma_db.query(
            query_embeddings=[combined_query_vector],
            n_results=k
        )
        
        # ENHANCED: Check for emotional memory seeds and prioritize them
        emotional_seeds = []
        regular_memories = []
        
        if results and results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                synopsis = metadata.get('synopsis', 'No synopsis available')
                
                # Check if this is an emotional memory seed
                if metadata.get('node_type') == 'custom' and metadata.get('importance_score', 0) > 0.7:
                    # This is likely an emotional memory seed
                    emotional_seeds.append({
                        'index': i,
                        'synopsis': synopsis,
                        'metadata': metadata,
                        'importance': metadata.get('importance_score', 0),
                        'personality_influence': metadata.get('personality_influence', 0)
                    })
                else:
                    # Regular memory
                    regular_memories.append({
                        'index': i,
                        'synopsis': synopsis,
                        'metadata': metadata
                    })
        
        # Helper to format timestamps for human-friendly display
        def _format_ts(ts: str) -> str:
            try:
                # Support both "+00:00" and trailing "Z"
                ts_clean = ts.replace('Z', '+00:00') if isinstance(ts, str) else ts
                dt = datetime.fromisoformat(ts_clean)
                # Normalize to UTC label
                dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                return dt_utc.strftime('%Y-%m-%d %H:%M UTC')
            except Exception:
                return 'Unknown time'

        # Build context with emotional seeds prioritized
        contexts = []
        
        # Add emotional seeds first (up to 2)
        for seed in sorted(emotional_seeds, key=lambda x: x['importance'], reverse=True)[:2]:
            metadata = seed['metadata']
            synopsis = seed['synopsis']
            ts_display = _format_ts(metadata.get('timestamp', ''))
            
            # Enhanced context info for emotional seeds
            context_info = f"[{ts_display}] [🎭 Emotional Memory: importance={seed['importance']:.2f}, influence={seed['personality_influence']:.2f}] {synopsis}"
            contexts.append(context_info)
            
            logger.debug(f"🎭 Retrieved emotional seed: {synopsis[:50]}...")
        
        # Add regular memories
        for memory in regular_memories[:k-len(contexts)]:
            metadata = memory['metadata']
            synopsis = memory['synopsis']
            ts_display = _format_ts(metadata.get('timestamp', ''))
            
            # Add affect information if available - but keep it clean for user-facing prompts
            if metadata.get('dual_channel'):
                user_affect_mag = metadata.get('user_affect_magnitude', 0)
                self_affect_mag = metadata.get('self_affect_magnitude', 0)
                # Store technical info but don't include in user-facing context
                # context_info = f"[Dual-affect: user={user_affect_mag:.2f}, self={self_affect_mag:.2f}] {synopsis}"
                context_info = f"[{ts_display}] {synopsis}"  # Clean context for prompts with timestamp
            else:
                affect_mag = sum(abs(x) for x in q_affect) if q_affect else 0
                # context_info = f"[Affect: {affect_mag:.2f}] {synopsis}"
                context_info = f"[{ts_display}] {synopsis}"  # Clean context for prompts with timestamp
            
            contexts.append(context_info)
        
        # ENHANCED: Try to get additional emotional seeds if not enough regular context
        if len(contexts) < k // 2:  # If we have less than half the requested context
            try:
                from .emotional_seed_enhancement import emotional_seed_enhancement
                
                # Validate inputs before additional retrieval
                validated_affect = q_affect if isinstance(q_affect, list) and len(q_affect) == 28 else [0.0] * 28
                
                additional_seeds = await emotional_seed_enhancement.get_seeds_for_context(
                    "general", validated_affect
                )
                
                # Validate additional seeds
                if isinstance(additional_seeds, list):
                    existing_seed_ids = {s['metadata'].get('node_id') for s in emotional_seeds if isinstance(s, dict)}
                    
                    for seed_node in additional_seeds[:2]:  # Add up to 2 more seeds
                        try:
                            if (hasattr(seed_node, 'node_id') and 
                                seed_node.node_id not in existing_seed_ids and
                                hasattr(seed_node, 'synopsis') and
                                hasattr(seed_node, 'importance_score')):
                                
                                importance_score = getattr(seed_node, 'importance_score', 0)
                                synopsis = getattr(seed_node, 'synopsis', 'No synopsis')
                                
                                if isinstance(importance_score, (int, float)) and isinstance(synopsis, str):
                                    context_info = f"[🎭 Context Seed: {importance_score:.2f}] {synopsis[:100]}"
                                    contexts.append(context_info)
                                    
                        except (AttributeError, TypeError) as e:
                            logger.debug(f"Error processing additional seed: {e}")
                            continue
                        
            except Exception as e:
                logger.debug(f"Could not retrieve additional emotional seeds: {e}")
        
        logger.info(f"📚 Retrieved {len(contexts)} context memories (seeds: {len(emotional_seeds)}, regular: {len(regular_memories)}) for query: '{query[:50]}...'")
        return contexts
    
    except Exception as e:
        logger.error(f"❌ Error retrieving context: {e}")
        return []

async def retrieve_context_with_compression(query: str, q_affect: list[float], messages: list[Message]) -> list[str]:
    """Retrieve context with intelligent compression based on conversation length"""
    # Base retrieval count
    base_k = 10
    
    # Adjust retrieval count based on conversation length
    conversation_length = len(messages)
    if conversation_length > 20:
        k = max(5, base_k - (conversation_length - 20) // 5)  # Reduce retrieval as conversation grows
    else:
        k = base_k
    
    # Retrieve contexts
    contexts = await retrieve_context(query, q_affect, k)
    
    # Apply compression if conversation is getting long
    if conversation_length > 15 and len(contexts) > 5:
        compressed_contexts = await compress_context_memories(contexts)
        logger.info(f"🗜️ Compressed {len(contexts)} contexts to {len(compressed_contexts)} for long conversation")
        return compressed_contexts
    
    return contexts

async def compress_context_memories(contexts: list[str]) -> list[str]:
    """Compress context memories by grouping similar themes"""
    if len(contexts) <= 5:
        return contexts
    
    # Simple compression: group by affect levels and take most relevant
    high_affect = []
    medium_affect = []
    low_affect = []
    
    for context in contexts:
        # Extract affect level from context string
        if "user=" in context or "self=" in context:
            # Parse dual-affect context
            try:
                if "user=" in context:
                    user_part = context.split("user=")[1].split(",")[0]
                    user_affect = float(user_part)
                    if user_affect > 1.0:
                        high_affect.append(context)
                    elif user_affect > 0.5:
                        medium_affect.append(context)
                    else:
                        low_affect.append(context)
                else:
                    medium_affect.append(context)
            except:
                medium_affect.append(context)
        else:
            # Parse single affect context
            try:
                affect_str = context.split("[Affect: ")[1].split("]")[0]
                affect_val = float(affect_str)
                if affect_val > 2.0:
                    high_affect.append(context)
                elif affect_val > 1.0:
                    medium_affect.append(context)
                else:
                    low_affect.append(context)
            except:
                medium_affect.append(context)
    
    # Select balanced representation
    compressed = []
    compressed.extend(high_affect[:2])   # Top 2 high-affect memories
    compressed.extend(medium_affect[:2]) # Top 2 medium-affect memories
    compressed.extend(low_affect[:1])    # Top 1 low-affect memory
    
    return compressed if compressed else contexts[:5]  # Fallback to first 5

async def get_recent_memories(limit: int = 50):
    """Get recent memories from the database"""
    try:
        from ..config import chroma_db, neo4j_conn
        
        # Try Neo4j first for better timestamp sorting
        if neo4j_conn:
            try:
                with neo4j_conn.session() as session:
                    result = session.run("""
                        MATCH (n)
                        WHERE n.timestamp IS NOT NULL
                        RETURN n.id as id, n.content as content, n.synopsis as synopsis,
                               n.timestamp as timestamp, n.origin as origin,
                               n.user_affect_magnitude as user_affect_magnitude,
                               n.self_affect_magnitude as self_affect_magnitude,
                               n.reflection as reflection
                        ORDER BY n.timestamp DESC
                        LIMIT $limit
                    """, limit=limit * 2)  # Get more to ensure we capture recent ones
                    
                    memories = []
                    for record in result:
                        memory = {
                            "id": record["id"] or "unknown",
                            "content": (record["content"] or "")[:200] + "..." if len(record["content"] or "") > 200 else (record["content"] or ""),
                            "synopsis": record["synopsis"] or "No synopsis",
                            "timestamp": record["timestamp"] or "Unknown",
                            "origin": record["origin"] or "unknown",
                            "has_reflection": bool(record.get("reflection")) if record.get("reflection") is not None else False
                        }
                        
                        # Add affect information
                        user_affect_mag = record["user_affect_magnitude"] or 0
                        self_affect_mag = record["self_affect_magnitude"] or 0
                        
                        # Ensure values are numeric
                        try:
                            user_affect_mag = float(user_affect_mag) if user_affect_mag is not None else 0.0
                            self_affect_mag = float(self_affect_mag) if self_affect_mag is not None else 0.0
                        except (ValueError, TypeError):
                            user_affect_mag = 0.0
                            self_affect_mag = 0.0
                        
                        memory["user_affect_magnitude"] = user_affect_mag
                        memory["self_affect_magnitude"] = self_affect_mag
                        
                        # Determine type based on affect vectors
                        if self_affect_mag > 0:
                            memory["type"] = "dual_affect"
                        else:
                            memory["type"] = "single_affect"
                        
                        memories.append(memory)
                    
                    # Limit to requested amount after sorting
                    memories = memories[:limit]
                    
                    logger.debug(f"📚 Retrieved {len(memories)} memories from Neo4j (sorted by timestamp)")
                    
                    return {
                        "memories": memories,
                        "total_count": len(memories),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
            except Exception as neo_error:
                logger.warning(f"Neo4j retrieval failed, falling back to ChromaDB: {neo_error}")
        
        # Fallback to ChromaDB with improved retrieval
        if not chroma_db:
            return {"error": "Database not initialized"}
        
        # Get MORE memories from ChromaDB to ensure we capture recent ones
        retrieval_limit = max(limit * 3, 150)  # Get 3x more to account for storage order
        results = chroma_db.get(
            limit=retrieval_limit,
            include=['metadatas', 'documents']
        )
        
        memories = []
        if results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas']):
                content = results['documents'][i] if i < len(results['documents']) else ""
                
                memory = {
                    "id": metadata.get('node_id', 'unknown'),
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "synopsis": metadata.get('synopsis', 'No synopsis'),
                    "timestamp": metadata.get('timestamp', 'Unknown'),
                    "origin": metadata.get('origin', 'unknown'),
                    "has_reflection": metadata.get('has_reflection', False)
                }
                
                # Add affect information
                user_affect_mag = metadata.get('user_affect_magnitude', 0)
                self_affect_mag = metadata.get('self_affect_magnitude', 0)
                
                # Ensure values are numeric
                try:
                    user_affect_mag = float(user_affect_mag) if user_affect_mag is not None else 0.0
                    self_affect_mag = float(self_affect_mag) if self_affect_mag is not None else 0.0
                except (ValueError, TypeError):
                    user_affect_mag = 0.0
                    self_affect_mag = 0.0
                
                memory["user_affect_magnitude"] = user_affect_mag
                memory["self_affect_magnitude"] = self_affect_mag
                
                if metadata.get('dual_channel'):
                    memory["type"] = "dual_affect"
                else:
                    memory["type"] = "single_affect"
                
                memories.append(memory)
        
        # Sort memories by timestamp (most recent first)
        try:
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            logger.debug(f"📚 Sorted {len(memories)} memories by timestamp")
        except Exception as e:
            logger.warning(f"Could not sort memories by timestamp: {e}")
            # If sorting fails, leave them in original order
        
        # Limit to requested amount after sorting
        memories = memories[:limit]
        
        return {
            "memories": memories,
            "total_count": len(memories),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting recent memories: {e}")
        return {"error": str(e)}

async def get_recent_echoes(limit: int = 20):
    """Get recent echo updates from Neo4j"""
    try:
        from ..config import neo4j_conn
        if not neo4j_conn:
            return {"error": "Neo4j not initialized"}
        
        with neo4j_conn.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.timestamp IS NOT NULL
                RETURN n.id as id, n.timestamp as timestamp, n.synopsis as synopsis,
                       n.affect_magnitude as affect_magnitude, n.origin as origin
                ORDER BY n.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            echoes = []
            for record in result:
                echoes.append({
                    "id": record["id"],
                    "timestamp": record["timestamp"],
                    "synopsis": record["synopsis"],
                    "affect_magnitude": record["affect_magnitude"],
                    "origin": record["origin"]
                })
            
            return {
                "echoes": echoes,
                "total_count": len(echoes),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error getting recent echoes: {e}")
        return {"error": str(e)}

async def get_context_for_query(query: str, user_affect: list[float] = None, limit: int = 10):
    """Get context memories for a specific query"""
    try:
        if not user_affect:
            user_affect = [0.0] * 28  # Default empty affect vector
        
        contexts = await retrieve_context(query, user_affect, limit)
        
        return {
            "query": query,
            "contexts": contexts,
            "context_count": len(contexts),
            "user_affect_magnitude": sum(abs(x) for x in user_affect),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting context for query: {e}")
        return {"error": str(e)}

async def get_memory_stats():
    """Get statistics about the memory system"""
    try:
        from ..config import chroma_db, neo4j_conn
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # ChromaDB stats
        if chroma_db:
            try:
                chroma_results = chroma_db.get(limit=1)
                chroma_count = chroma_db.count() if hasattr(chroma_db, 'count') else len(chroma_results.get('ids', []))
                stats["chroma_memories"] = chroma_count
            except:
                stats["chroma_memories"] = "unavailable"
        else:
            stats["chroma_memories"] = "not_initialized"
            
        # Neo4j stats
        if neo4j_conn:
            try:
                with neo4j_conn.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    stats["neo4j_nodes"] = result.single()["count"]
            except:
                stats["neo4j_nodes"] = "unavailable"
        else:
            stats["neo4j_nodes"] = "not_initialized"
            
        return stats
    
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"error": str(e)} 