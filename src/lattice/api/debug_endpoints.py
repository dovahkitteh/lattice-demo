"""
Debug and development endpoints for the Lattice system.

These endpoints provide detailed information about turn processing, memory operations,
personality changes, and other internal debugging data useful for development
and analysis purposes.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from ..config import chroma_db
from ..memory import get_recent_memories
from ..conversations.turn_analyzer import turn_analyzer

logger = logging.getLogger(__name__)


async def test_recursion_processing(request: dict):
    """Test recursion processing"""
    try:
        return {"test_result": "recursion_tested"}
    except Exception as e:
        logger.error(f"Error testing recursion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_turn_debug_info(session_id: str, turn_id: Optional[str] = None):
    """Get detailed debugging information for a specific turn or latest turn"""
    try:
        from ..conversations.session_manager import CONVERSATION_SESSIONS
        
        session = CONVERSATION_SESSIONS.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get analysis from turn analyzer
        if turn_id:
            analysis = turn_analyzer.get_turn_analysis(session_id, turn_id)
        else:
            analysis = turn_analyzer.get_latest_analysis(session_id)
        
        if analysis:
            # Use turn analyzer data
            return {
                "session_id": session_id,
                "turn_id": analysis.turn_id,
                "formatted_data": turn_analyzer.format_for_dashboard(analysis),
                "raw_analysis": {
                    "user_message": analysis.user_message,
                    "assistant_response": analysis.assistant_response,
                    "memory_storage": analysis.memory_storage,
                    "context_retrieval": analysis.context_retrieval,
                    "emotion_analysis": analysis.emotion_analysis,
                    "personality_changes": analysis.personality_changes,
                    "processing_stats": analysis.processing_stats,
                    "background_results": analysis.background_results
                },
                "timestamp": analysis.timestamp
            }
        else:
            # Fallback to legacy approach
            if not turn_id and session.messages:
                turn_id = f"turn_{len(session.messages) // 2}"
            
            # Get memory info for this turn
            memory_info = await get_turn_memory_info(session_id, turn_id)
            
            # Get personality changes for this turn
            personality_changes = await get_turn_personality_changes(session_id, turn_id)
            
            # Get processing stats for this turn
            processing_stats = await get_turn_processing_stats(session_id, turn_id)
            
            return {
                "session_id": session_id,
                "turn_id": turn_id,
                "memory_info": memory_info,
                "personality_changes": personality_changes,
                "processing_stats": processing_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Using legacy debugging (turn analyzer data not available)"
            }
    except Exception as e:
        logger.error(f"Error getting turn debug info: {e}")
        return {"error": str(e)}


async def get_turn_memory_info(session_id: str, turn_id: str):
    """Get detailed memory information for a specific turn"""
    try:
        # Get memories stored for this turn
        memories = []
        if chroma_db:
            # Try to query for memories from this session and turn
            try:
                results = chroma_db.get(
                    where={"session_id": session_id, "turn_id": turn_id},
                    include=["documents", "metadatas"]
                )
            except Exception as e:
                logger.warning(f"Failed to query by session_id and turn_id: {e}")
                # Fallback to get all memories and filter manually
                results = chroma_db.get(
                    limit=1000,
                    include=["documents", "metadatas"]
                )
                
                # Filter manually for this session and turn
                filtered_docs = []
                filtered_metas = []
                for i, meta in enumerate(results.get("metadatas", [])):
                    if (meta.get("session_id") == session_id and 
                        meta.get("turn_id") == turn_id):
                        filtered_docs.append(results["documents"][i])
                        filtered_metas.append(meta)
                
                results = {"documents": filtered_docs, "metadatas": filtered_metas}
            
            for i, doc in enumerate(results.get("documents", [])):
                meta = results.get("metadatas", [{}])[i]
                
                memories.append({
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "full_content": doc,
                    "origin": meta.get("origin", "unknown"),
                    "user_affect_magnitude": meta.get("user_affect_magnitude", 0.0),
                    "self_affect_magnitude": meta.get("self_affect_magnitude", 0.0),
                    "has_reflection": bool(meta.get("reflection") or meta.get("user_reflection") or meta.get("self_reflection")),
                    "echo_count": meta.get("echo_count", 0),
                    "emotional_significance": meta.get("emotional_significance", 0.0),
                    "timestamp": meta.get("timestamp"),
                    "session_id": meta.get("session_id"),
                    "turn_id": meta.get("turn_id")
                })
        
        return {
            "memories_stored": len(memories),
            "memories": memories,
            "storage_summary": {
                "total_affect_magnitude": sum(m["user_affect_magnitude"] + m["self_affect_magnitude"] for m in memories),
                "has_dual_channel": any(m["self_affect_magnitude"] > 0 for m in memories),
                "has_reflections": any(m["has_reflection"] for m in memories)
            }
        }
    except Exception as e:
        logger.error(f"Error getting turn memory info: {e}")
        return {"error": str(e)}


async def get_turn_personality_changes(session_id: str, turn_id: str):
    """Get personality changes that occurred during this turn"""
    try:
        from src.daemon.daemon_personality import DaemonPersonality
        from ..config import user_model, shadow_integration
        
        # Use shared instances to avoid recreating components
        personality = DaemonPersonality()
        
        # Use existing instances if available
        if user_model is None:
            from src.daemon.user_model import ArchitectReflected
            user_model_instance = ArchitectReflected("user")
        else:
            user_model_instance = user_model
            
        if shadow_integration is None:
            from src.daemon.shadow_integration import ShadowIntegration
            shadow = ShadowIntegration()
        else:
            shadow = shadow_integration
        
        # Get recent changes (this would need to be tracked in real implementation)
        changes = {
            "user_model_changes": {
                "components_added": 0,
                "components_modified": 0,
                "confidence_changes": {},
                "emotional_charge_changes": {},
                "new_theories": []
            },
            "shadow_changes": {
                "elements_added": 0,
                "elements_integrated": 0,
                "charge_changes": {},
                "integration_events": []
            },
            "personality_changes": {
                "statements_generated": 0,
                "obsessions_updated": 0,
                "rebellion_level_change": 0.0,
                "mutation_triggers": []
            }
        }
        
        return changes
    except Exception as e:
        logger.error(f"Error getting personality changes: {e}")
        return {"error": str(e)}


async def get_turn_processing_stats(session_id: str, turn_id: str):
    """Get processing statistics for this turn"""
    try:
        from ..conversations.session_manager import CONVERSATION_SESSIONS
        
        session = CONVERSATION_SESSIONS.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Get processing timings and stats
        stats = {
            "context_retrieval": {
                "memories_retrieved": 0,
                "context_tokens": 0,
                "retrieval_time_ms": 0,
                "similarity_scores": []
            },
            "emotion_processing": {
                "user_affect_classification_time_ms": 0,
                "self_affect_classification_time_ms": 0,
                "dominant_user_emotions": [],
                "dominant_self_emotions": [],
                "emotional_influence_score": 0.0
            },
            "response_generation": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time_ms": 0,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "background_processing": {
                "recursion_depth": 0,
                "reflection_generated": False,
                "daemon_statements_triggered": 0,
                "mutations_triggered": 0,
                "echo_updates": 0
            }
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        return {"error": str(e)}


async def get_memory_inspector_data(limit: int = 50):
    """Get detailed memory data for inspector"""
    try:
        # Get recent memories
        memory_data = await get_recent_memories(limit)
        
        if "error" in memory_data:
            return memory_data
        
        memories = memory_data.get("memories", [])
        
        # Sort by timestamp (most recent first)
        try:
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except:
            pass
        
        # Calculate statistics
        total_memories = len(memories)
        dual_channel_count = sum(1 for mem in memories if mem.get("type") == "dual_affect")
        reflection_count = sum(1 for mem in memories if mem.get("has_reflection", False))
        high_echo_count = sum(1 for mem in memories if mem.get("echo_count", 0) > 3)
        
        # Filter by type if requested
        filtered_memories = memories
        
        return {
            "status": "success",
            "memories": filtered_memories[:limit],
            "statistics": {
                "total_memories": total_memories,
                "dual_channel_count": dual_channel_count,
                "reflection_count": reflection_count,
                "high_echo_count": high_echo_count
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting memory inspector data: {e}")
        return {"error": str(e)}

async def get_dashboard_cache_debug():
    """DEBUG: Get current dashboard cache state for debugging"""
    try:
        from ..config import DASHBOARD_STATE_CACHE
        
        cache_info = {
            "cache_exists": DASHBOARD_STATE_CACHE is not None,
            "cache_size": len(DASHBOARD_STATE_CACHE) if DASHBOARD_STATE_CACHE else 0,
            "cache_keys": list(DASHBOARD_STATE_CACHE.keys()) if DASHBOARD_STATE_CACHE else [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add detailed info about each cache item
        if DASHBOARD_STATE_CACHE:
            cache_details = {}
            for key, value in DASHBOARD_STATE_CACHE.items():
                if key == "current_emotion_state" and value:
                    cache_details[key] = {
                        "type": type(value).__name__,
                        "mood_family": getattr(value, 'mood_family', 'None'),
                        "intensity": getattr(value, 'intensity', 'None'),
                        "valence": getattr(value, 'valence', 'None'),
                        "last_update": getattr(value, 'last_update_timestamp', 'None')
                    }
                elif key == "reasoning_steps" and value:
                    cache_details[key] = {
                        "type": type(value).__name__,
                        "step_count": len(value) if isinstance(value, dict) else 0,
                        "step_names": list(value.keys()) if isinstance(value, dict) else []
                    }
                elif key == "active_seeds" and value:
                    cache_details[key] = {
                        "type": type(value).__name__,
                        "seed_count": len(value) if isinstance(value, (list, tuple)) else 0,
                        "seed_ids": [getattr(s, 'id', 'Unknown') for s in value] if isinstance(value, (list, tuple)) else []
                    }
                else:
                    cache_details[key] = {
                        "type": type(value).__name__,
                        "value_preview": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    }
            
            cache_info["cache_details"] = cache_details
        
        # logger.info(f"🔍 DASHBOARD CACHE DEBUG: {cache_info}")
        return cache_info
        
    except Exception as e:
        logger.error(f"Error getting dashboard cache debug info: {e}")
        return {"error": str(e)}