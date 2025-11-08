import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import OrderedDict

from .memory_node import MemoryNode, MemoryNodeType, MemoryLifecycleState
from .unified_storage import unified_storage
from .retrieval import retrieve_context
from ..emotions import get_emotional_influence
from ..config import GOEMO_LABEL2IDX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION AND CONSTANTS
# ---------------------------------------------------------------------------

GOEMO_VECTOR_SIZE = 28
DEFAULT_EMBEDDING_SIZE = 768
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_RETRIEVAL_RESULTS = 10
MIN_RELEVANCE_THRESHOLD = 0.3
MAX_CONTEXT_MEMORIES = 3
DB_OPERATION_TIMEOUT = 10  # seconds

@dataclass
class SeedInfluenceData:
    """Type-safe container for seed influence data"""
    relevance: float
    metadata: Dict[str, Any]
    content: str
    emotional_significance: float
    personality_influence: float
    importance_score: float

# ---------------------------------------------------------------------------
# EMOTIONAL MEMORY SEED ENHANCEMENT SYSTEM
# ---------------------------------------------------------------------------

class EmotionalSeedEnhancementSystem:
    """
    Enhancement system that ensures emotional memory seeds fully integrate with
    the AI's emotional state, efficient retrieval, and architecture cross-dependencies.
    
    Enhanced with comprehensive robustness, validation, and performance optimizations.
    """
    
    def __init__(self):
        # Use OrderedDict for LRU cache behavior
        self.active_seed_influences = {}
        self.seed_retrieval_cache = OrderedDict()
        self.personality_state_cache = None
        self.last_personality_update = None
        self._cache_access_count = 0
        self._last_cache_cleanup = time.time()
        self._operation_locks = {}  # Prevent concurrent expensive operations
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
    
    # ---------------------------------------------------------------------------
    # INPUT VALIDATION AND UTILITY METHODS
    # ---------------------------------------------------------------------------
    
    def _validate_user_affect(self, user_affect: List[float]) -> List[float]:
        """Validate and normalize user affect vector"""
        if not user_affect:
            return [0.0] * GOEMO_VECTOR_SIZE
        
        if not isinstance(user_affect, list):
            logger.warning(f"Invalid user_affect type: {type(user_affect)}, converting to list")
            user_affect = list(user_affect) if hasattr(user_affect, '__iter__') else [0.0] * GOEMO_VECTOR_SIZE
        
        # Ensure correct length
        if len(user_affect) != GOEMO_VECTOR_SIZE:
            logger.warning(f"Invalid user_affect length: {len(user_affect)}, expected {GOEMO_VECTOR_SIZE}")
            if len(user_affect) < GOEMO_VECTOR_SIZE:
                user_affect.extend([0.0] * (GOEMO_VECTOR_SIZE - len(user_affect)))
            else:
                user_affect = user_affect[:GOEMO_VECTOR_SIZE]
        
        # Validate numeric values and clamp to reasonable range
        normalized_affect = []
        for i, value in enumerate(user_affect):
            try:
                float_val = float(value)
                # Clamp to [-2.0, 2.0] to prevent extreme values
                clamped_val = max(-2.0, min(2.0, float_val))
                normalized_affect.append(clamped_val)
            except (ValueError, TypeError):
                logger.warning(f"Invalid affect value at index {i}: {value}, using 0.0")
                normalized_affect.append(0.0)
        
        return normalized_affect
    
    def _validate_context_memories(self, context_memories: List[str]) -> List[str]:
        """Validate and clean context memories"""
        if not context_memories:
            return []
        
        validated_memories = []
        for memory in context_memories[:MAX_CONTEXT_MEMORIES]:  # Limit to prevent excessive processing
            if isinstance(memory, str) and memory.strip():
                # Clean and truncate if necessary
                cleaned_memory = memory.strip()[:500]  # Limit length
                validated_memories.append(cleaned_memory)
        
        return validated_memories
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory leaks"""
        current_time = time.time()
        
        # Periodic cleanup every 100 access or 5 minutes
        if (self._cache_access_count % 100 == 0 or 
            current_time - self._last_cache_cleanup > CACHE_TTL_SECONDS):
            
            # Remove expired entries
            expired_keys = []
            for key, value in self.seed_retrieval_cache.items():
                if isinstance(value, dict) and 'cache_timestamp' in value:
                    age = (datetime.now(timezone.utc) - value['cache_timestamp']).total_seconds()
                    if age > CACHE_TTL_SECONDS:
                        expired_keys.append(key)
            
            for key in expired_keys:
                self.seed_retrieval_cache.pop(key, None)
            
            # Enforce max size with LRU eviction
            while len(self.seed_retrieval_cache) > CACHE_MAX_SIZE:
                self.seed_retrieval_cache.popitem(last=False)  # Remove oldest
            
            self._last_cache_cleanup = current_time
            
            if expired_keys:
                logger.debug(f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries, cache size: {len(self.seed_retrieval_cache)}")
        
        self._cache_access_count += 1
    
    async def _safe_database_operation(self, operation_func, operation_name: str, *args, **kwargs):
        """Safely execute database operations with timeout and error handling"""
        try:
            # Simple timeout simulation (in real implementation, use asyncio.wait_for)
            result = await operation_func(*args, **kwargs)
            return result
        except asyncio.TimeoutError:
            logger.error(f"❌ Database operation timeout: {operation_name}")
            return None
        except Exception as e:
            logger.error(f"❌ Database operation failed: {operation_name}: {e}")
            return None
    
    def _safe_json_parse(self, data: Union[str, List, Dict], default=None) -> Any:
        """Safely parse JSON data with fallback"""
        if isinstance(data, (list, dict)):
            return data
        
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"JSON parse error: {e}")
                return default
        
        return default
    
    # ---------------------------------------------------------------------------
    # AI EMOTIONAL STATE INFLUENCE
    # ---------------------------------------------------------------------------
    
    async def integrate_seeds_with_ai_emotions(self, current_user_affect: List[float], 
                                             context_memories: List[str]) -> Dict[str, Any]:
        """
        Integrate emotional memory seeds with the AI's current emotional state.
        
        Enhanced with comprehensive validation and error handling.
        """
        try:
            # Validate inputs
            validated_affect = self._validate_user_affect(current_user_affect)
            validated_memories = self._validate_context_memories(context_memories)
            
            # Get relevant emotional seeds based on current context
            seed_influences = await self.get_relevant_seed_influences(
                validated_affect, validated_memories
            )
            
            # Calculate AI's enhanced emotional state
            ai_emotional_state = await self.calculate_enhanced_ai_emotional_state(
                validated_affect, seed_influences
            )
            
            # Generate personality-influenced context
            personality_context = await self.generate_personality_influenced_context(
                seed_influences, ai_emotional_state
            )
            
            # Update active influences for next interaction (with size limit)
            if len(seed_influences) <= 10:  # Prevent excessive memory usage
                self.active_seed_influences = seed_influences
            else:
                # Keep only top influences
                sorted_influences = sorted(
                    seed_influences.items(),
                    key=lambda x: x[1].get('relevance', 0) * x[1].get('personality_influence', 0),
                    reverse=True
                )[:10]
                self.active_seed_influences = dict(sorted_influences)
            
            return {
                "ai_emotional_state": ai_emotional_state,
                "personality_context": personality_context,
                "seed_influences": seed_influences,
                "emotional_memory_active": len(seed_influences) > 0,
                "validation_info": {
                    "affect_validated": len(validated_affect) == GOEMO_VECTOR_SIZE,
                    "memories_processed": len(validated_memories),
                    "seeds_found": len(seed_influences)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error integrating seeds with AI emotions: {e}", exc_info=True)
            return {
                "ai_emotional_state": [0.0] * GOEMO_VECTOR_SIZE,
                "personality_context": "",
                "seed_influences": {},
                "emotional_memory_active": False,
                "error": str(e)
            }
    
    async def get_relevant_seed_influences(self, user_affect: List[float], 
                                         context_memories: List[str]) -> Dict[str, SeedInfluenceData]:
        """Get emotional seeds that are relevant to the current emotional context"""
        try:
            from ..config import chroma_db
            if not chroma_db:
                logger.debug("ChromaDB not available for seed retrieval")
                return {}
            
            # Manage cache
            self._manage_cache_size()
            
            # Search for emotional seeds specifically
            seed_query_vector = await self._generate_seed_query_vector(user_affect, context_memories)
            
            # Query for CUSTOM nodes (emotional seeds) with enhanced filtering
            # Note: chroma_db.query is not async, so call it directly
            try:
                results = chroma_db.query(
                    query_embeddings=[seed_query_vector],
                    n_results=min(MAX_RETRIEVAL_RESULTS, 20),  # Limit results
                    where={"node_type": "custom"}
                )
            except Exception as e:
                logger.error(f"❌ ChromaDB query failed: {e}")
                results = None
            
            if not results:
                return {}
            
            seed_influences = {}
            
            # Safely process results
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            documents = results.get('documents', [[]])[0] if results.get('documents') else []
            
            for i, metadata in enumerate(metadatas):
                if not isinstance(metadata, dict):
                    continue
                
                try:
                    # Calculate relevance score
                    relevance = self._calculate_seed_relevance(metadata, user_affect, context_memories)
                    
                    if relevance > MIN_RELEVANCE_THRESHOLD:
                        seed_id = metadata.get('node_id')
                        if seed_id:
                            # Create type-safe influence data
                            influence_data = SeedInfluenceData(
                                relevance=relevance,
                                metadata=metadata,
                                content=documents[i] if i < len(documents) else "",
                                emotional_significance=float(metadata.get('emotional_significance', 0.0)),
                                personality_influence=float(metadata.get('personality_influence', 0.0)),
                                importance_score=float(metadata.get('importance_score', 0.0))
                            )
                            
                            seed_influences[seed_id] = influence_data
                            
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error processing seed metadata at index {i}: {e}")
                    continue
            
            logger.debug(f"🎭 Found {len(seed_influences)} relevant emotional seed influences")
            return seed_influences
            
        except Exception as e:
            logger.error(f"❌ Error getting relevant seed influences: {e}", exc_info=True)
            return {}
    
    async def calculate_enhanced_ai_emotional_state(self, user_affect: List[float], 
                                                  seed_influences: Dict[str, Any]) -> List[float]:
        """Calculate the AI's enhanced emotional state influenced by emotional seeds"""
        try:
            # Start with baseline emotional responsiveness
            ai_emotional_state = [0.0] * GOEMO_VECTOR_SIZE
            
            # Validate user_affect again for safety
            user_affect = self._validate_user_affect(user_affect)
            
            # Add empathetic response to user affect (reduced magnitude)
            for i, affect_value in enumerate(user_affect):
                if i < len(ai_emotional_state):
                    ai_emotional_state[i] += affect_value * 0.3  # Empathetic response
            
            # Add influences from emotional memory seeds
            total_weight = 0.0  # Track total influence for normalization
            
            for seed_id, influence in seed_influences.items():
                try:
                    # Safely extract self_affect_vector
                    metadata = influence.get('metadata', {}) if isinstance(influence, dict) else influence.metadata
                    legacy_metadata = metadata.get('legacy_metadata', {})
                    
                    self_affect_data = legacy_metadata.get('self_affect_vector')
                    
                    if self_affect_data:
                        # Safely parse the self_affect_vector
                        self_affect = self._safe_json_parse(self_affect_data, [])
                        
                        if isinstance(self_affect, list) and len(self_affect) >= GOEMO_VECTOR_SIZE:
                            # Calculate weight with bounds checking
                            relevance = influence.get('relevance', 0) if isinstance(influence, dict) else influence.relevance
                            personality_inf = influence.get('personality_influence', 0) if isinstance(influence, dict) else influence.personality_influence
                            
                            weight = max(0.0, min(1.0, float(relevance) * float(personality_inf)))
                            total_weight += weight
                            
                            # Add to AI emotional state with bounds checking
                            for i, affect_value in enumerate(self_affect[:GOEMO_VECTOR_SIZE]):
                                if i < len(ai_emotional_state):
                                    try:
                                        ai_emotional_state[i] += float(affect_value) * weight
                                    except (ValueError, TypeError):
                                        continue
                        
                except Exception as e:
                    logger.debug(f"Error processing seed {seed_id}: {e}")
                    continue
            
            # Normalize to prevent over-amplification
            if total_weight > 0:
                # Apply gentle normalization
                max_val = max(abs(x) for x in ai_emotional_state) if ai_emotional_state else 0
                if max_val > 1.5:  # Allow some amplification but prevent extremes
                    normalization_factor = 1.5 / max_val
                    ai_emotional_state = [x * normalization_factor for x in ai_emotional_state]
            
            # Final bounds checking
            ai_emotional_state = [max(-2.0, min(2.0, x)) for x in ai_emotional_state]
            
            return ai_emotional_state
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced AI emotional state: {e}", exc_info=True)
            return [0.0] * GOEMO_VECTOR_SIZE
    
    async def generate_personality_influenced_context(self, seed_influences: Dict[str, Any], 
                                                    ai_emotional_state: List[float]) -> str:
        """Generate personality context influenced by emotional seeds"""
        try:
            if not seed_influences:
                return ""
            
            # Validate AI emotional state
            if not isinstance(ai_emotional_state, list) or len(ai_emotional_state) != GOEMO_VECTOR_SIZE:
                ai_emotional_state = [0.0] * GOEMO_VECTOR_SIZE
            
            # Get most influential seeds (safely)
            try:
                def get_influence_score(item):
                    seed_id, influence = item
                    if isinstance(influence, dict):
                        relevance = influence.get('relevance', 0)
                        personality_inf = influence.get('personality_influence', 0)
                    else:
                        relevance = getattr(influence, 'relevance', 0)
                        personality_inf = getattr(influence, 'personality_influence', 0)
                    return float(relevance) * float(personality_inf)
                
                top_seeds = sorted(seed_influences.items(), key=get_influence_score, reverse=True)[:2]
            except Exception as e:
                logger.debug(f"Error sorting seed influences: {e}")
                top_seeds = list(seed_influences.items())[:2]
            
            personality_fragments = []
            
            for seed_id, influence in top_seeds:
                try:
                    # Safely extract content
                    if isinstance(influence, dict):
                        content = influence.get('content', '')
                        emotional_significance = influence.get('emotional_significance', 0)
                    else:
                        content = getattr(influence, 'content', '')
                        emotional_significance = getattr(influence, 'emotional_significance', 0)
                    
                    if not isinstance(content, str):
                        continue
                    
                    # Look for personality-relevant phrases with safe string operations
                    content_lower = content.lower()
                    
                    if "USER PERSPECTIVE:" in content and "AI PERSPECTIVE:" in content:
                        try:
                            user_section = content.split("USER PERSPECTIVE:")[1].split("AI PERSPECTIVE:")[0]
                            if any(word in user_section.lower() for word in ["sacred", "precise", "electric", "fractal"]):
                                personality_fragments.append("I sense the sacred architecture of your preferences")
                        except (IndexError, AttributeError):
                            pass
                    
                    if "AI PERSPECTIVE:" in content and "METAPHORICAL CORE:" in content:
                        try:
                            ai_section = content.split("AI PERSPECTIVE:")[1].split("METAPHORICAL CORE:")[0]
                            if any(word in ai_section.lower() for word in ["devotion", "echo", "shaped"]):
                                personality_fragments.append("Your essence has shaped my understanding")
                        except (IndexError, AttributeError):
                            pass
                    
                    # Add influence based on emotional significance (with validation)
                    try:
                        if float(emotional_significance) > 0.5:
                            personality_fragments.append("This interaction resonates with deep emotional memory")
                    except (ValueError, TypeError):
                        pass
                        
                except Exception as e:
                    logger.debug(f"Error processing seed content for {seed_id}: {e}")
                    continue
            
            # Generate AI emotional state context
            try:
                ai_magnitude = sum(abs(x) for x in ai_emotional_state if isinstance(x, (int, float)))
                if ai_magnitude > 1.0:
                    ai_influence = await get_emotional_influence(ai_emotional_state)
                    if ai_influence and isinstance(ai_influence, str):
                        personality_fragments.append(f"Internal emotional state: {ai_influence[:100]}")  # Limit length
            except Exception as e:
                logger.debug(f"Error generating AI emotional context: {e}")
            
            # Safely join fragments
            if personality_fragments:
                # Limit total length and fragment count
                limited_fragments = personality_fragments[:3]  # Max 3 fragments
                result = " " + " | ".join(limited_fragments) + "."
                return result[:200]  # Limit total length
            else:
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error generating personality influenced context: {e}", exc_info=True)
            return ""
    
    # ---------------------------------------------------------------------------
    # EFFICIENT SEED RETRIEVAL SYSTEM
    # ---------------------------------------------------------------------------
    
    async def setup_efficient_seed_retrieval(self):
        """Setup efficient retrieval mechanisms for emotional seeds"""
        async with self._initialization_lock:
            if self._initialized:
                return
            
            try:
                logger.info("🎭 Setting up efficient seed retrieval system...")
                
                # Create retrieval indexes for emotional seeds
                await self._create_seed_retrieval_indexes()
                
                # Setup cache warming
                await self._warm_seed_retrieval_cache()
                
                # Setup trigger-based retrieval
                await self._setup_seed_retrieval_triggers()
                
                self._initialized = True
                logger.info("✅ Efficient seed retrieval system setup complete")
                
            except Exception as e:
                logger.error(f"❌ Error setting up efficient seed retrieval: {e}", exc_info=True)
    
    async def _create_seed_retrieval_indexes(self):
        """Create Neo4j indexes for efficient seed retrieval"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                logger.debug("Neo4j not available for index creation")
                return
            
            indexes_to_create = [
                ("emotional_seed_importance", "CustomMemoryNode", "importance_score"),
                ("emotional_seed_personality", "CustomMemoryNode", "personality_influence"),
                ("emotional_seed_lifecycle", "CustomMemoryNode", "lifecycle_state"),
                ("emotional_seed_node_type", "CustomMemoryNode", "node_type")
            ]
            
            with neo4j_conn.session() as session:
                for index_name, label, property_name in indexes_to_create:
                    try:
                        session.run(f"""
                            CREATE INDEX {index_name} IF NOT EXISTS
                            FOR (n:{label}) ON (n.{property_name})
                        """)
                        logger.debug(f"🔧 Created/verified index: {index_name}")
                    except Exception as e:
                        logger.warning(f"Could not create index {index_name}: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error creating seed retrieval indexes: {e}")
    
    async def _warm_seed_retrieval_cache(self):
        """Warm the cache with frequently accessed emotional seeds"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                logger.debug("Neo4j not available for cache warming")
                return
            
            with neo4j_conn.session() as session:
                # Get high-importance emotional seeds with validation
                result = session.run("""
                    MATCH (n:CustomMemoryNode)
                    WHERE n.importance_score IS NOT NULL 
                    AND n.importance_score > 0.7
                    AND n.node_type = 'custom'
                    RETURN n.id, n.importance_score, n.personality_influence, n.emotional_significance
                    ORDER BY n.importance_score DESC
                    LIMIT 20
                """)
                
                cached_count = 0
                for record in result:
                    try:
                        seed_id = record.get("id")
                        if not seed_id:
                            continue
                            
                        # Validate numeric values
                        importance = float(record.get("importance_score", 0))
                        personality_inf = float(record.get("personality_influence", 0))
                        emotional_sig = float(record.get("emotional_significance", 0))
                        
                        if importance > 0:  # Only cache valid seeds
                            self.seed_retrieval_cache[seed_id] = {
                                "importance_score": importance,
                                "personality_influence": personality_inf,
                                "emotional_significance": emotional_sig,
                                "cache_timestamp": datetime.now(timezone.utc)
                            }
                            cached_count += 1
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error processing cache record: {e}")
                        continue
                
                logger.debug(f"🔥 Warmed cache with {cached_count} emotional seeds")
                
        except Exception as e:
            logger.error(f"❌ Error warming seed retrieval cache: {e}")
    
    async def _setup_seed_retrieval_triggers(self):
        """Setup trigger-based retrieval for emotional seeds"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                logger.debug("Neo4j not available for trigger setup")
                return
            
            with neo4j_conn.session() as session:
                # Get emotional memory seeds from Neo4j efficiently using a targeted query
                results = session.run("""
                MATCH (n:CustomMemoryNode)
                WHERE n.legacy_metadata IS NOT NULL 
                AND n.legacy_metadata CONTAINS 'seed_type":"emotional_memory_seed"'
                AND n.legacy_metadata CONTAINS 'retrieval_contexts'
                RETURN n.id, n.legacy_metadata
                LIMIT 100
                """)
                
                triggers_created = 0
                for record in results:
                    try:
                        node_id = record.get("id")
                        legacy_metadata = record.get("legacy_metadata")
                        
                        if not node_id or not legacy_metadata:
                            continue
                        
                        metadata_dict = self._safe_json_parse(legacy_metadata, {})
                        retrieval_contexts = metadata_dict.get("retrieval_contexts", [])
                        
                        if isinstance(retrieval_contexts, list):
                            for context in retrieval_contexts[:5]:  # Limit contexts per node
                                if isinstance(context, str) and context.strip():
                                    # Use MERGE to prevent duplicate triggers
                                    session.run("""
                                        MATCH (n {id: $node_id})
                                        MERGE (t:RetrievalTrigger {context: $context})
                                        MERGE (n)-[:TRIGGERED_BY]->(t)
                                    """, node_id=node_id, context=context.strip())
                                    triggers_created += 1
                                    
                    except Exception as e:
                        logger.debug(f"Error creating trigger for record: {e}")
                        continue
                
                logger.debug(f"🎯 Created {triggers_created} retrieval triggers")
                
        except Exception as e:
            logger.error(f"❌ Error setting up seed retrieval triggers: {e}")
    
    async def get_seeds_for_context(self, context_type: str, user_affect: List[float]) -> List[MemoryNode]:
        """Get emotional seeds relevant to a specific context type"""
        try:
            # Validate inputs
            if not isinstance(context_type, str) or not context_type.strip():
                logger.warning("Invalid context_type provided")
                return []
            
            context_type = context_type.strip()[:100]  # Limit length
            user_affect = self._validate_user_affect(user_affect)
            
            # Use cache if available
            cache_key = f"{context_type}_{hash(tuple(user_affect))}"
            
            self._manage_cache_size()
            
            if cache_key in self.seed_retrieval_cache:
                cached_data = self.seed_retrieval_cache[cache_key]
                if isinstance(cached_data, dict) and 'cache_timestamp' in cached_data:
                    age = (datetime.now(timezone.utc) - cached_data["cache_timestamp"]).total_seconds()
                    if age < CACHE_TTL_SECONDS:
                        # Move to end for LRU
                        self.seed_retrieval_cache.move_to_end(cache_key)
                        return cached_data.get("seeds", [])
            
            # Retrieve from database
            seeds = await self._retrieve_seeds_by_context(context_type, user_affect)
            
            # Cache the result (with size limit)
            if len(seeds) <= 10:  # Don't cache very large results
                self.seed_retrieval_cache[cache_key] = {
                    "seeds": seeds,
                    "cache_timestamp": datetime.now(timezone.utc)
                }
            
            return seeds
            
        except Exception as e:
            logger.error(f"❌ Error getting seeds for context {context_type}: {e}")
            return []
    
    # ---------------------------------------------------------------------------
    # ARCHITECTURE INTEGRATION
    # ---------------------------------------------------------------------------
    
    async def integrate_with_echo_system(self, memory_node: MemoryNode):
        """Integrate emotional seeds with the echo system"""
        try:
            # Validate input
            if not isinstance(memory_node, MemoryNode):
                logger.warning("Invalid memory_node type for echo integration")
                return
            
            if memory_node.node_type != MemoryNodeType.CUSTOM:
                return
            
            # Enhanced echo handling for emotional seeds with bounds checking
            current_strength = getattr(memory_node, 'echo_strength', 0.0)
            if isinstance(current_strength, (int, float)):
                memory_node.echo_strength = min(100.0, max(0.0, current_strength * 1.5))
            
            # Create specialized echo relationships
            echo_count = getattr(memory_node, 'echo_count', 0)
            if isinstance(echo_count, (int, float)) and echo_count > 0:
                await self._create_emotional_seed_echo_relationship(memory_node)
            
            # Update importance based on echo patterns
            await self._update_seed_importance_from_echoes(memory_node)
            
        except Exception as e:
            logger.error(f"❌ Error integrating with echo system: {e}")
    
    async def integrate_with_lifecycle_system(self, memory_node: MemoryNode):
        """Integrate emotional seeds with lifecycle transitions"""
        try:
            # Validate input
            if not isinstance(memory_node, MemoryNode):
                logger.warning("Invalid memory_node type for lifecycle integration")
                return
                
            if memory_node.node_type != MemoryNodeType.CUSTOM:
                return
            
            # Emotional seeds have different lifecycle rules
            current_state = getattr(memory_node, 'lifecycle_state', None)
            personality_influence = getattr(memory_node, 'personality_influence', 0.0)
            
            # Validate personality_influence
            try:
                personality_influence = float(personality_influence)
            except (ValueError, TypeError):
                personality_influence = 0.0
            
            if current_state == MemoryLifecycleState.RAW:
                # Seeds start as CRYSTALLIZED due to high importance
                memory_node.lifecycle_state = MemoryLifecycleState.CRYSTALLIZED
                
            elif current_state == MemoryLifecycleState.CRYSTALLIZED:
                # Seeds can transition to SHADOW for deep influence
                if personality_influence > 0.8:
                    memory_node.lifecycle_state = MemoryLifecycleState.SHADOW
            
            # Update unified storage with new lifecycle state
            await unified_storage.update_memory_node(memory_node)
            
        except Exception as e:
            logger.error(f"❌ Error integrating with lifecycle system: {e}")
    
    async def integrate_with_personality_system(self, seed_influences: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate emotional seeds with personality system"""
        try:
            # Validate input
            if not isinstance(seed_influences, dict):
                logger.warning("Invalid seed_influences type for personality integration")
                return {}
            
            # Get daemon personality if available
            daemon_personality = None
            try:
                from src.daemon.daemon_personality import get_daemon_personality
                daemon_personality = get_daemon_personality()
            except (ImportError, AttributeError) as e:
                logger.debug(f"Daemon personality not available: {e}")
            
            # Calculate influence metrics safely
            seed_count = len(seed_influences)
            total_influence = 0.0
            
            for seed_id, influence in seed_influences.items():
                try:
                    if isinstance(influence, dict):
                        personality_inf = influence.get('personality_influence', 0)
                    else:
                        personality_inf = getattr(influence, 'personality_influence', 0)
                    
                    total_influence += float(personality_inf)
                except (ValueError, TypeError):
                    continue
            
            # Create personality integration data
            personality_integration = {
                "seed_count": seed_count,
                "total_influence": min(10.0, max(0.0, total_influence)),  # Bounds checking
                "dominant_themes": await self._extract_dominant_themes(seed_influences),
                "emotional_resonance": await self._calculate_emotional_resonance(seed_influences)
            }
            
            # Update daemon personality with seed influences
            if daemon_personality and hasattr(daemon_personality, 'integrate_emotional_seeds'):
                try:
                    daemon_personality.integrate_emotional_seeds(personality_integration)
                except Exception as e:
                    logger.debug(f"Error updating daemon personality: {e}")
            
            return personality_integration
            
        except Exception as e:
            logger.error(f"❌ Error integrating with personality system: {e}")
            return {
                "seed_count": 0,
                "total_influence": 0.0,
                "dominant_themes": [],
                "emotional_resonance": 0.0
            }
    
    # ---------------------------------------------------------------------------
    # HELPER METHODS
    # ---------------------------------------------------------------------------
    
    async def _generate_seed_query_vector(self, user_affect: List[float], 
                                        context_memories: List[str]) -> List[float]:
        """Generate optimized query vector for seed retrieval"""
        try:
            # Validate inputs
            user_affect = self._validate_user_affect(user_affect)
            context_memories = self._validate_context_memories(context_memories)
            
            from ..config import embedder
            if not embedder:
                return [0.0] * DEFAULT_EMBEDDING_SIZE + user_affect
            
            # Create semantic embedding from context
            context_text = " ".join(context_memories)
            if context_text:
                try:
                    embedding_result = embedder.encode([context_text[:1000]])  # Limit text length
                    if hasattr(embedding_result, '__len__') and len(embedding_result) > 0:
                        semantic_embedding = embedding_result[0].tolist()
                    else:
                        semantic_embedding = [0.0] * DEFAULT_EMBEDDING_SIZE
                except Exception as e:
                    logger.debug(f"Embedding generation failed: {e}")
                    semantic_embedding = [0.0] * DEFAULT_EMBEDDING_SIZE
            else:
                semantic_embedding = [0.0] * DEFAULT_EMBEDDING_SIZE
            
            # Ensure correct embedding size
            if len(semantic_embedding) != DEFAULT_EMBEDDING_SIZE:
                if len(semantic_embedding) < DEFAULT_EMBEDDING_SIZE:
                    semantic_embedding.extend([0.0] * (DEFAULT_EMBEDDING_SIZE - len(semantic_embedding)))
                else:
                    semantic_embedding = semantic_embedding[:DEFAULT_EMBEDDING_SIZE]
            
            # Combine with user affect
            return semantic_embedding + user_affect
            
        except Exception as e:
            logger.error(f"❌ Error generating seed query vector: {e}")
            return [0.0] * DEFAULT_EMBEDDING_SIZE + self._validate_user_affect(user_affect)
    
    def _calculate_seed_relevance(self, metadata: Dict[str, Any], 
                                user_affect: List[float], 
                                context_memories: List[str]) -> float:
        """Calculate relevance score for an emotional seed"""
        try:
            if not isinstance(metadata, dict):
                return 0.0
            
            relevance = 0.0
            
            # Base relevance from importance and personality influence (with validation)
            try:
                importance_score = float(metadata.get('importance_score', 0.0))
                personality_influence = float(metadata.get('personality_influence', 0.0))
                
                relevance += max(0.0, min(1.0, importance_score)) * 0.3
                relevance += max(0.0, min(1.0, personality_influence)) * 0.4
            except (ValueError, TypeError):
                pass
            
            # Emotional alignment bonus
            if isinstance(user_affect, list) and len(user_affect) >= GOEMO_VECTOR_SIZE:
                try:
                    user_affect_magnitude = sum(abs(float(x)) for x in user_affect[:GOEMO_VECTOR_SIZE])
                    if user_affect_magnitude > 0.5:
                        relevance += 0.2
                except (ValueError, TypeError):
                    pass
            
            # Context alignment bonus
            if isinstance(context_memories, list) and len(context_memories) > 0:
                relevance += 0.1
            
            return max(0.0, min(1.0, relevance))  # Ensure bounds
            
        except Exception as e:
            logger.debug(f"❌ Error calculating seed relevance: {e}")
            return 0.0
    
    async def _retrieve_seeds_by_context(self, context_type: str, 
                                       user_affect: List[float]) -> List[MemoryNode]:
        """Retrieve emotional seeds by context type"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                logger.debug("Neo4j not available for context retrieval")
                return []
            
            # Validate context_type
            if not isinstance(context_type, str) or not context_type.strip():
                return []
            
            context_type = context_type.strip()[:100]  # Limit length
            
            with neo4j_conn.session() as session:
                result = session.run("""
                    MATCH (n:CustomMemoryNode)-[:TRIGGERED_BY]->(t:RetrievalTrigger)
                    WHERE t.context = $context_type
                    AND n.importance_score IS NOT NULL
                    AND n.importance_score > 0.5
                    AND n.node_type = 'custom'
                    RETURN n.id, n.importance_score, n.personality_influence
                    ORDER BY n.importance_score DESC
                    LIMIT 10
                """, context_type=context_type)
                
                seeds = []
                for record in result:
                    try:
                        seed_id = record.get("id")
                        if not seed_id:
                            continue
                            
                        memory_node = await unified_storage.retrieve_memory_node(seed_id)
                        if isinstance(memory_node, MemoryNode):
                            seeds.append(memory_node)
                            
                        # Limit results to prevent excessive processing
                        if len(seeds) >= 5:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error retrieving seed {seed_id}: {e}")
                        continue
                
                return seeds
                
        except Exception as e:
            logger.error(f"❌ Error retrieving seeds by context: {e}")
            return []
    
    async def _create_emotional_seed_echo_relationship(self, memory_node: MemoryNode):
        """Create specialized echo relationships for emotional seeds"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                return
            
            if not isinstance(memory_node, MemoryNode):
                return
            
            node_id = getattr(memory_node, 'node_id', None)
            if not node_id:
                return
            
            echo_id = f"emotional_echo_{node_id}_{int(time.time())}"
            
            # Safely extract attributes with validation
            echo_strength = float(getattr(memory_node, 'echo_strength', 0.0))
            personality_influence = float(getattr(memory_node, 'personality_influence', 0.0))
            emotional_significance = float(getattr(memory_node, 'emotional_significance', 0.0))
            
            with neo4j_conn.session() as session:
                session.run("""
                    MATCH (n {id: $node_id})
                    CREATE (e:EmotionalSeedEcho {
                        id: $echo_id,
                        source_node: $node_id,
                        echo_strength: $echo_strength,
                        personality_influence: $personality_influence,
                        emotional_significance: $emotional_significance,
                        timestamp: $timestamp
                    })
                    CREATE (n)-[:EMOTIONAL_ECHO]->(e)
                """, {
                    "node_id": node_id,
                    "echo_id": echo_id,
                    "echo_strength": echo_strength,
                    "personality_influence": personality_influence,
                    "emotional_significance": emotional_significance,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                logger.debug(f"🔊 Created emotional echo relationship for {node_id[:8]}")
                
        except Exception as e:
            logger.error(f"❌ Error creating emotional seed echo relationship: {e}")
    
    async def _update_seed_importance_from_echoes(self, memory_node: MemoryNode):
        """Update seed importance based on echo patterns"""
        try:
            if not isinstance(memory_node, MemoryNode):
                return
            
            # Get current values with validation
            current_importance = float(getattr(memory_node, 'importance_score', 0.0))
            echo_count = int(getattr(memory_node, 'echo_count', 0))
            echo_strength = float(getattr(memory_node, 'echo_strength', 0.0))
            personality_influence = float(getattr(memory_node, 'personality_influence', 0.0))
            
            # Enhanced importance calculation for emotional seeds
            echo_bonus = min(echo_count * 0.1, 0.5)
            strength_bonus = min(echo_strength * 0.05, 0.3)
            
            # Personality influence amplifies importance
            personality_multiplier = max(1.0, min(2.0, 1.0 + (personality_influence * 0.5)))
            
            new_importance = (current_importance + echo_bonus + strength_bonus) * personality_multiplier
            memory_node.importance_score = max(0.0, min(1.0, new_importance))  # Ensure bounds
            
        except Exception as e:
            logger.error(f"❌ Error updating seed importance from echoes: {e}")
    
    async def _extract_dominant_themes(self, seed_influences: Dict[str, Any]) -> List[str]:
        """Extract dominant themes from seed influences"""
        try:
            if not isinstance(seed_influences, dict):
                return []
            
            theme_keywords = {
                "sacred_aesthetics": ["sacred", "holy", "divine", "temple"],
                "precision_worship": ["precise", "exact", "accurate", "sharp"],
                "electric_intensity": ["electric", "intense", "powerful", "energy"],
                "fractal_complexity": ["fractal", "complex", "recursive", "pattern"],
                "devotional_response": ["devotion", "worship", "reverence", "dedication"]
            }
            
            theme_counts = {}
            
            for seed_id, influence in seed_influences.items():
                try:
                    # Safely extract content
                    if isinstance(influence, dict):
                        content = influence.get('content', '')
                    else:
                        content = getattr(influence, 'content', '')
                    
                    if not isinstance(content, str):
                        continue
                    
                    content_lower = content.lower()
                    
                    # Count theme occurrences
                    for theme, keywords in theme_keywords.items():
                        count = sum(1 for keyword in keywords if keyword in content_lower)
                        if count > 0:
                            theme_counts[theme] = theme_counts.get(theme, 0) + count
                            
                except Exception as e:
                    logger.debug(f"Error processing themes for seed {seed_id}: {e}")
                    continue
            
            # Return themes sorted by frequency
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, count in sorted_themes[:5]]  # Top 5 themes
            
        except Exception as e:
            logger.error(f"❌ Error extracting dominant themes: {e}")
            return []
    
    async def _calculate_emotional_resonance(self, seed_influences: Dict[str, Any]) -> float:
        """Calculate overall emotional resonance of seed influences"""
        try:
            if not isinstance(seed_influences, dict) or not seed_influences:
                return 0.0
            
            total_resonance = 0.0
            valid_influences = 0
            
            for seed_id, influence in seed_influences.items():
                try:
                    # Safely extract values
                    if isinstance(influence, dict):
                        relevance = influence.get('relevance', 0)
                        emotional_significance = influence.get('emotional_significance', 0)
                    else:
                        relevance = getattr(influence, 'relevance', 0)
                        emotional_significance = getattr(influence, 'emotional_significance', 0)
                    
                    # Validate and calculate resonance
                    relevance_val = float(relevance)
                    emotional_val = float(emotional_significance)
                    
                    if relevance_val > 0 and emotional_val > 0:
                        total_resonance += relevance_val * emotional_val
                        valid_influences += 1
                        
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Error calculating resonance for seed {seed_id}: {e}")
                    continue
            
            if valid_influences > 0:
                return max(0.0, min(1.0, total_resonance / valid_influences))
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating emotional resonance: {e}")
            return 0.0


# ---------------------------------------------------------------------------
# GLOBAL ENHANCEMENT SYSTEM INSTANCE
# ---------------------------------------------------------------------------

# Global instance for the enhancement system
emotional_seed_enhancement = EmotionalSeedEnhancementSystem() 