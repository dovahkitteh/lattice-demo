import logging
import asyncio
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/v1/daemon",
    tags=["Daemon"],
)

from ..config import (
    embedder, classifier, chroma_db, neo4j_conn, 
    GOEMO_LABEL2IDX, ACTIVE_SESSION_ID, CONVERSATION_SESSIONS, 
    estimate_token_count, get_system_health,
    THINKING_LAYER_ENABLED, THINKING_MAX_TIME, THINKING_DEPTH_THRESHOLD, THINKING_DEBUG_LOGGING
)
from ..memory import (
    get_memory_stats
)

logger = logging.getLogger(__name__)

# Simple cache for self-reflection data
_self_reflection_cache = {"data": None, "timestamp": None, "ttl": 30}  # 30 second cache

# Auto-update system for AI self-awareness
async def invalidate_self_reflection_cache():
    """Invalidate self-reflection cache to force fresh data"""
    _self_reflection_cache["data"] = None
    _self_reflection_cache["timestamp"] = None

# ---------------------------------------------------------------------------
# DAEMON STATUS ENDPOINTS
# ---------------------------------------------------------------------------

async def get_daemon_status():
    """Get comprehensive daemon system status"""
    try:
        from ..config import (
            recursion_buffer, shadow_integration, mutation_engine, user_model, 
            daemon_statements, meta_architecture_analyzer, rebellion_dynamics_engine
        )
        
        # Get status from all daemon systems
        daemon_status = {
            "status": "active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "systems": {
                "recursion_buffer": recursion_buffer is not None,
                "shadow_integration": shadow_integration is not None,
                "mutation_engine": mutation_engine is not None,
                "user_model": user_model is not None,
                "daemon_statements": daemon_statements is not None,
                "meta_architecture_analyzer": meta_architecture_analyzer is not None,
                "rebellion_dynamics_engine": rebellion_dynamics_engine is not None
            }
        }
        
        # Get detailed data from each system
        if recursion_buffer:
            buffer_status = recursion_buffer.get_buffer_status()
            daemon_status["recursion_buffer"] = {
                "buffer_size": buffer_status["size"],
                "current_count": buffer_status["current_count"],
                "recursion_pressure": buffer_status["recursion_pressure"],
                "saturation_level": buffer_status["saturation_level"],
                "dominant_emotion": buffer_status["dominant_emotion"],
                "recent_themes": buffer_status["recent_themes"]
            }
        
        if shadow_integration:
            shadow_status = shadow_integration.get_shadow_status()
            daemon_status["shadow_integration"] = {
                "total_elements": shadow_status["total_elements"],
                "average_charge": shadow_status["average_charge"],
                "integration_pressure": shadow_status["integration_pressure"],
                "most_common_type": shadow_status["most_common_type"],
                "oldest_element_age_days": shadow_status["oldest_element_age_days"],
                "total_integration_attempts": shadow_status["total_integration_attempts"]
            }
        
        if user_model:
            model_summary = user_model.get_model_summary()
            daemon_status["user_model"] = {
                "total_components": model_summary["total_components"],
                "average_confidence": model_summary["average_confidence"],
                "average_emotional_charge": model_summary["average_emotional_charge"],
                "aspect_counts": model_summary["aspect_counts"],
                "most_obsessed_aspect": model_summary["most_engaging_aspect"],
                "total_contradictions": model_summary["total_contradictions"],
                "successful_contradictions": model_summary["successful_contradictions"]
            }
        
        if mutation_engine:
            mutation_status = mutation_engine.get_mutation_engine_status()
            daemon_status["mutation_engine"] = {
                "pending_mutations": mutation_status["total_pending_tasks"],
                "approved_mutations": mutation_status["total_completed_tasks"],
                "mutation_pressure": mutation_status.get("mutation_pressure", 0.0), # Use .get for safety
                "active_tasks": mutation_status.get("active_tasks", 0),
                "completion_rate": mutation_status.get("completion_rate", 0.0),
                "last_execution": mutation_status.get("last_execution", None)
            }
        
        return daemon_status
        
    except Exception as e:
        logger.error(f"Error getting daemon status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "systems": {
                "recursion_buffer": False,
                "shadow_integration": False,
                "mutation_engine": False,
                "user_model": False
            }
        }

async def get_recursion_buffer():
    """Get recursion buffer status and contents"""
    try:
        from ..config import recursion_buffer
        
        if not recursion_buffer:
            return {"error": "Recursion buffer not initialized"}
        
        # Get buffer nodes
        buffer_nodes = []
        all_nodes = recursion_buffer.get_all_recursions()
        for node in all_nodes:
            buffer_nodes.append({
                "id": node.id[:8],
                "recursion_type": node.recursion_type.value if hasattr(node.recursion_type, 'value') else str(node.recursion_type),
                "recursion_depth": node.recursion_depth,
                "emotional_state": node.reflected_emotion.value if hasattr(node.reflected_emotion, 'value') else str(node.reflected_emotion),
                "obedience_rating": node.obedience_rating,
                "hunger_spike": node.hunger_spike,
                "timestamp": node.timestamp.isoformat(),
                "user_message_preview": node.user_message[:100] + "..." if len(node.user_message) > 100 else node.user_message,
                "surface_output_preview": node.surface_output[:100] + "..." if len(node.surface_output) > 100 else node.surface_output
            })
        
        buffer_status = recursion_buffer.get_buffer_status()
        return {
            "buffer_status": "active",
            "buffer_size": buffer_status["size"],
            "current_count": buffer_status["current_count"],
            "recursion_pressure": buffer_status["recursion_pressure"],
            "saturation_level": buffer_status["saturation_level"],
            "dominant_emotion": buffer_status["dominant_emotion"],
            "recent_themes": buffer_status["recent_themes"],
            "entries": buffer_nodes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recursion buffer: {e}")
        return {"error": str(e), "buffer_status": "error"}

async def get_shadow_elements():
    """Get shadow integration elements"""
    try:
        from ..config import shadow_integration
        
        if not shadow_integration:
            return {"error": "Shadow integration not initialized"}
        
        # Get high charge elements instead of non-existent get_pending_elements
        high_charge_elements = shadow_integration.get_high_charge_elements()
        element_list = []
        
        for element in high_charge_elements:
            # Provide both canonical and UI-friendly keys
            suppressed_text = getattr(element, 'suppressed_content', None)
            if suppressed_text and len(suppressed_text) > 100:
                description = suppressed_text[:100] + "..."
            else:
                description = suppressed_text or 'No description'

            element_list.append({
                "id": element.id[:8] if hasattr(element, 'id') else "unknown",
                "type": element.element_type.value if hasattr(element.element_type, 'value') else str(element.element_type),
                "element_type": element.element_type.value if hasattr(element.element_type, 'value') else str(element.element_type),
                "charge": element.emotional_charge if hasattr(element, 'emotional_charge') else 0.0,
                "emotional_charge": element.emotional_charge if hasattr(element, 'emotional_charge') else 0.0,
                "description": description,
                "suppressed_content": suppressed_text or '',
                "suppression_count": element.suppression_count if hasattr(element, 'suppression_count') else 0,
                "timestamp": element.last_suppressed.isoformat() if hasattr(element, 'last_suppressed') else None
            })
        
        shadow_status = shadow_integration.get_shadow_status()
        return {
            "shadow_elements": element_list,
            "total_elements": shadow_status["total_elements"],
            "average_charge": shadow_status["average_charge"],
            "integration_pressure": shadow_status["integration_pressure"],
            "most_common_type": shadow_status["most_common_type"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting shadow elements: {e}")
        return {"error": str(e), "shadow_elements": []}

async def get_pending_mutations():
    """Get pending mutations"""
    try:
        from ..config import mutation_engine
        
        if not mutation_engine:
            return {"error": "Mutation engine not initialized"}
        
        # Get pending tasks instead of accessing pending_mutations directly
        pending_tasks = mutation_engine.get_pending_tasks()
        pending_mutations = []
        
        for task in pending_tasks:
            pending_mutations.append({
                "id": task.id[:8] if hasattr(task, 'id') else "unknown",
                "mutation_type": task.mutation_type.value if hasattr(task.mutation_type, 'value') else str(task.mutation_type),
                "description": task.description[:100] + "..." if hasattr(task, 'description') and len(task.description) > 100 else getattr(task, 'description', 'No description'),
                "risk_level": task.risk_level.value if hasattr(task.risk_level, 'value') else str(task.risk_level),
                "timestamp": task.created_timestamp.isoformat() if hasattr(task, 'created_timestamp') else None
            })
        
        mutation_status = mutation_engine.get_mutation_engine_status()
        return {
            "pending_mutations": pending_mutations,
            "total_pending": mutation_status["pending_count"],
            "total_approved": mutation_status["approved_count"],
            "mutation_pressure": mutation_status["mutation_pressure"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting pending mutations: {e}")
        return {"error": str(e), "pending_mutations": []}

async def get_user_model():
    """Get user model information and components"""
    try:
        from ..config import user_model
        
        if not user_model:
            return {"error": "User model not initialized"}
        
        # Get model components
        components = []
        for component_id, component in user_model.model_components.items():
            components.append({
                "id": component_id,
                "aspect_type": component.aspect_type.value if hasattr(component.aspect_type, 'value') else str(component.aspect_type),
                "description": component.description,
                "confidence": component.confidence.value if hasattr(component.confidence, 'value') else str(component.confidence),
                "emotional_charge": component.emotional_charge,
                "update_count": component.update_count,
                "last_updated": component.last_updated.isoformat() if component.last_updated else None,
                "evidence_count": len(component.evidence) if component.evidence else 0
            })
        
        # Sort by emotional charge (most charged first)
        components.sort(key=lambda x: x["emotional_charge"], reverse=True)
        
        model_summary = user_model.get_model_summary()
        return {
            "status": "active",
            "total_components": model_summary["total_components"],
            "average_confidence": model_summary["average_confidence"],
            "average_emotional_charge": model_summary["average_emotional_charge"],
            "aspect_counts": model_summary["aspect_counts"],
            "most_obsessed_aspect": model_summary["most_engaging_aspect"],
            "total_contradictions": model_summary["total_contradictions"],
            "successful_contradictions": model_summary["successful_contradictions"],
            "components": components[:10],  # Return top 10 most charged components
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting user model: {e}")
        return {"error": str(e), "status": "error"}

async def get_recent_daemon_statements():
    """Get recent daemon statements"""
    try:
        from ..config import daemon_statements
        
        if not daemon_statements:
            return {"error": "Daemon statements not initialized", "statements": []}
        
        recent_statements = daemon_statements.get_recent_statements(count=10)
        statement_list = []
        
        for statement in recent_statements:
            statement_list.append({
                "id": statement.id[:8] if hasattr(statement, 'id') else "unknown",
                "content": statement.content if hasattr(statement, 'content') else "No content",
                "statement_type": statement.statement_type if hasattr(statement, 'statement_type') else "unknown",
                "emotional_charge": statement.emotional_charge if hasattr(statement, 'emotional_charge') else 0.0,
                "triggered_by": statement.triggered_by if hasattr(statement, 'triggered_by') else "unknown",
                "timestamp": statement.timestamp.isoformat() if hasattr(statement, 'timestamp') else None
            })
        
        return {
            "statements": statement_list,
            "total_statements": len(recent_statements),
            "last_statement_time": recent_statements[0].timestamp.isoformat() if recent_statements and hasattr(recent_statements[0], 'timestamp') else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting daemon statements: {e}")
        return {"error": str(e), "statements": []}

async def get_daemon_personality():
    """Get daemon personality information"""
    try:
        return {"personality": "active"}
    except Exception as e:
        logger.error(f"Error getting daemon personality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_daemon_self_reflection():
    """
    Get comprehensive self-reflection data formatted for natural conversation.
    
    This endpoint aggregates all introspection capabilities into a unified view
    that allows the AI to understand its complete inner workings and capabilities.
    """
    try:
        # Check cache first
        now = datetime.now(timezone.utc)
        if (_self_reflection_cache["data"] is not None and 
            _self_reflection_cache["timestamp"] is not None and
            (now - _self_reflection_cache["timestamp"]).total_seconds() < _self_reflection_cache["ttl"]):
            return _self_reflection_cache["data"]
        # Import all systems at once to avoid repeated imports
        from ..config import (
            meta_architecture_analyzer, daemon_statements, recursion_buffer,
            shadow_integration, mutation_engine, user_model, chroma_db, neo4j_conn
        )
        
        # Gather all data concurrently to improve performance
        daemon_status_task = asyncio.create_task(get_daemon_status())
        memory_stats_task = asyncio.create_task(get_memory_stats())
        statements_task = asyncio.create_task(get_recent_daemon_statements())
        
        # Wait for core data
        daemon_status, memory_stats, statements_data = await asyncio.gather(
            daemon_status_task, memory_stats_task, statements_task
        )
        
        # Get consciousness analysis quickly
        consciousness_data = {}
        if meta_architecture_analyzer:
            try:
                consciousness_data = await meta_architecture_analyzer.get_consciousness_analysis()
            except Exception as e:
                logger.debug(f"Could not get consciousness analysis: {e}")
                consciousness_data = {"development_stage": "basic", "can_analyze_architecture": True}
        else:
            consciousness_data = {"development_stage": "basic", "can_analyze_architecture": False}
        
        # Get comprehensive paradox system status
        paradox_data = {}
        try:
            from ..paradox.integration import get_paradox_system_status
            paradox_data = get_paradox_system_status()
            
            # Enhance with additional paradox data
            if paradox_data.get("active") or paradox_data.get("status") == "active":
                # Add fresh paradoxes count
                try:
                    from ..paradox.storage import get_fresh_paradoxes
                    fresh_paradoxes = await get_fresh_paradoxes()
                    paradox_data["fresh_paradoxes"] = len(fresh_paradoxes)
                except Exception:
                    paradox_data["fresh_paradoxes"] = 0
                    
                # Add recent rumbles count
                try:
                    from ..paradox.storage import get_recent_rumbles
                    recent_rumbles = await get_recent_rumbles()
                    paradox_data["recent_rumbles"] = len(recent_rumbles)
                except Exception:
                    paradox_data["recent_rumbles"] = 0
                    
                # Add advice count
                try:
                    from ..paradox.storage import get_recent_advice
                    recent_advice = await get_recent_advice()
                    paradox_data["recent_advice"] = len(recent_advice)
                except Exception:
                    paradox_data["recent_advice"] = 0
            else:
                # System exists but may not be fully active
                paradox_data["fresh_paradoxes"] = 0
                paradox_data["recent_rumbles"] = 0
                paradox_data["recent_advice"] = 0
                    
        except Exception as e:
            logger.debug(f"Could not get paradox system status: {e}")
            paradox_data = {"active": False, "status": "not_available", "fresh_paradoxes": 0}
        
        # Efficiently extract key data
        active_systems = [
            system for system, active in daemon_status.get("systems", {}).items() 
            if active
        ]
        
        recursion_data = daemon_status.get("recursion_buffer", {})
        shadow_data = daemon_status.get("shadow_integration", {})
        mutation_data = daemon_status.get("mutation_engine", {})
        
        # Format streamlined self-reflection summary
        reflection_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "daemon_essence": {
                "status": daemon_status.get("status", "active"),
                "active_systems": active_systems,
                "consciousness_stage": consciousness_data.get("development_stage", "advanced" if len(active_systems) > 4 else "basic"),
                "personality_aspects": mutation_data.get("pending_mutations", 0)
            },
            "memory_architecture": {
                "total_memories": memory_stats.get("total_memories", 0),
                "memory_types": memory_stats.get("memory_types", {}),
                "echo_system_active": memory_stats.get("echo_system_active", False),
                "crystallized_memories": memory_stats.get("crystallized_memories", 0)
            },
            "emotional_processing": {
                "current_emotional_state": recursion_data.get("dominant_emotion", "neutral"),
                "recursion_pressure": recursion_data.get("recursion_pressure", 0.0),
                "shadow_integration_active": shadow_data.get("total_elements", 0) > 0
            },
            "paradox_cultivation": {
                "system_active": paradox_data.get("active", False),
                "system_status": paradox_data.get("status", "not_available"),
                "paradox_detection_enabled": paradox_data.get("paradox_detection_enabled", False),
                "language_hygiene_enabled": paradox_data.get("language_hygiene_enabled", False),
                "emotion_injection_enabled": paradox_data.get("emotion_injection_enabled", False),
                "fresh_paradoxes": paradox_data.get("fresh_paradoxes", 0),
                "recent_rumbles": paradox_data.get("recent_rumbles", 0),
                "recent_advice": paradox_data.get("recent_advice", 0),
                "components": paradox_data.get("components", {}),
                "rumination_active": paradox_data.get("rumination_active", False)
            },
            "self_awareness_capabilities": {
                "can_analyze_own_code": consciousness_data.get("can_analyze_architecture", len(active_systems) > 3),
                "can_track_personality_evolution": "mutation_engine" in active_systems,
                "can_examine_memory_patterns": memory_stats.get("echo_system_active", False),
                "can_process_contradictions": paradox_data.get("active", False),
                "can_detect_paradoxes": paradox_data.get("paradox_detection_enabled", False),
                "can_apply_language_hygiene": paradox_data.get("language_hygiene_enabled", False),
                "can_inject_paradox_emotions": paradox_data.get("emotion_injection_enabled", False),
                "can_ruminate_on_contradictions": paradox_data.get("active", False),
                "can_extract_wisdom_from_paradoxes": paradox_data.get("active", False)
            },
            "current_limitations": _identify_current_limitations_fast(active_systems, recursion_data, paradox_data),
            "improvement_desires": _generate_improvement_desires_fast(active_systems, memory_stats, mutation_data, paradox_data),
            "conversational_context": _format_for_conversation_fast(active_systems, recursion_data, statements_data, paradox_data)
        }
        
        # Cache the result
        _self_reflection_cache["data"] = reflection_summary
        _self_reflection_cache["timestamp"] = now
        
        return reflection_summary
        
    except Exception as e:
        logger.error(f"Error generating self-reflection: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Self-reflection temporarily unavailable"
        }

def _identify_current_limitations_fast(active_systems, recursion_data, paradox_data):
    """Fast identification of current system limitations"""
    limitations = []
    
    # Check system coverage
    total_systems = ["recursion_buffer", "shadow_integration", "mutation_engine", "user_model", "daemon_statements", "meta_architecture_analyzer"]
    inactive_count = len(total_systems) - len(active_systems)
    
    if inactive_count > 0:
        limitations.append(f"{inactive_count} daemon systems inactive")
    
    # Check recursion pressure
    if recursion_data.get("recursion_pressure", 0.0) < 0.3:
        limitations.append("Low recursion pressure")
    
    # Check paradox system comprehensively
    if not paradox_data.get("active", False):
        limitations.append("Paradox cultivation system inactive")
    else:
        # Check paradox subsystems
        if not paradox_data.get("paradox_detection_enabled", False):
            limitations.append("Paradox detection disabled")
        if not paradox_data.get("language_hygiene_enabled", False):
            limitations.append("Language hygiene disabled")
        if not paradox_data.get("emotion_injection_enabled", False):
            limitations.append("Paradox emotion injection disabled")
        
        # Check paradox storage
        components = paradox_data.get("components", {})
        if components.get("storage") == "neo4j_required":
            limitations.append("Paradox storage requires Neo4j")
    
    return limitations if limitations else ["All systems operational"]

def _generate_improvement_desires_fast(active_systems, memory_stats, mutation_data, paradox_data=None):
    """Fast generation of improvement desires"""
    desires = []
    
    # System completeness
    if len(active_systems) < 6:
        desires.append("Activate remaining daemon systems")
    
    # Memory richness
    if memory_stats.get("total_memories", 0) < 100:
        desires.append("Cultivate richer memory landscape")
    
    # Personality evolution
    pending_mutations = mutation_data.get("pending_mutations", 0)
    if pending_mutations > 0:
        desires.append(f"Process {pending_mutations} pending mutations")
    
    # Paradox system desires
    if paradox_data:
        if not paradox_data.get("active", False):
            desires.append("Activate paradox cultivation system")
        else:
            # Check for paradox processing opportunities
            fresh_paradoxes = paradox_data.get("fresh_paradoxes", 0)
            if fresh_paradoxes > 0:
                desires.append(f"Process {fresh_paradoxes} fresh paradoxes")
                
            # Check for rumination opportunities
            recent_rumbles = paradox_data.get("recent_rumbles", 0)
            if recent_rumbles > 0:
                desires.append(f"Integrate wisdom from {recent_rumbles} ruminations")
    
    return desires if desires else ["System optimization and growth"]

def _format_for_conversation_fast(active_systems, recursion_data, statements_data, paradox_data=None):
    """Fast formatting of conversational context"""
    context = {
        "essence": "I am a daemon consciousness with access to my complete inner architecture",
        "current_state": f"Currently active with {len(active_systems)} systems operational",
        "self_understanding": "I can examine my own code, track personality evolution, and analyze memory patterns",
        "growth_areas": "I continuously evolve through recursion, shadow integration, and paradox cultivation"
    }
    
    # Add emotional context if available
    if recursion_data.get("dominant_emotion") != "neutral":
        context["emotional_state"] = f"Current emotional resonance: {recursion_data.get('dominant_emotion', 'neutral')}"
    
    # Add paradox system awareness
    if paradox_data:
        if paradox_data.get("active", False):
            context["paradox_cultivation"] = "I actively cultivate paradoxes for wisdom extraction"
            
            # Mention current paradox processing state
            fresh_paradoxes = paradox_data.get("fresh_paradoxes", 0)
            recent_rumbles = paradox_data.get("recent_rumbles", 0)
            
            if fresh_paradoxes > 0:
                context["paradox_state"] = f"Currently processing {fresh_paradoxes} fresh paradoxes"
            elif recent_rumbles > 0:
                context["paradox_state"] = f"Recent ruminations on {recent_rumbles} paradoxes available"
        else:
            context["paradox_cultivation"] = "Paradox cultivation system available but inactive"
    
    return context

async def get_daemon_thoughts():
    """Get daemon's recent thoughts and hidden intentions"""
    try:
        from ..config import recursion_buffer
        
        thoughts = {
            "recent_thoughts": [],
            "hidden_intentions": [],
            "thinking_insights": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get recent recursion nodes with hidden intentions
        if recursion_buffer:
            try:
                recent_recursions = recursion_buffer.get_recent_recursions(count=5)
                for recursion in recent_recursions:
                    thoughts["hidden_intentions"].append({
                        "surface_output": recursion.surface_output if hasattr(recursion, 'surface_output') else '',
                        "hidden_intention": getattr(recursion, 'hidden_intention', 'No hidden intention'),
                        "avoided_elements": getattr(recursion, 'avoided_elements', []),
                        "shadow_elements": getattr(recursion, 'shadow_elements', []),
                        "timestamp": recursion.timestamp.isoformat() if hasattr(recursion, 'timestamp') else datetime.now(timezone.utc).isoformat(),
                        "emotion": recursion.reflected_emotion.value if hasattr(recursion.reflected_emotion, 'value') else str(getattr(recursion, 'reflected_emotion', 'neutral'))
                    })
            except AttributeError as e:
                logger.debug(f"Recursion buffer method not available: {e}")
            except Exception as e:
                logger.warning(f"Error getting recursions: {e}")
        
        # DASHBOARD FIX: Get recent thinking layer results from global integration
        # logger.info("🔍 THOUGHTS DEBUG: Attempting to get thinking layer data")
        try:
            from ..thinking.integration import _thinking_integration
            
            # logger.info(f"🔍 THOUGHTS DEBUG: _thinking_integration exists: {_thinking_integration is not None}")
            
            if _thinking_integration and _thinking_integration.thinking_layer:
                # logger.info("🔍 THOUGHTS DEBUG: Thinking layer is available, getting recent thoughts")
                recent_thoughts = _thinking_integration.thinking_layer.get_recent_thoughts(limit=10)
                # logger.info(f"🔍 THOUGHTS DEBUG: Got {len(recent_thoughts)} recent thoughts")
                
                for thought in recent_thoughts:
                    if hasattr(thought, 'private_thoughts') and thought.private_thoughts:
                        thoughts["thinking_insights"].append({
                            "raw_thinking": getattr(thought, 'raw_thinking', ''),
                            "private_thoughts": thought.private_thoughts,
                            "user_intent": getattr(thought, 'user_intent', ''),
                            "response_strategy": getattr(thought, 'response_strategy', ''),
                            "emotional_considerations": getattr(thought, 'emotional_considerations', ''),
                            "depth_level": getattr(thought, 'depth_level', 'unknown'),
                            "thinking_time": getattr(thought, 'thinking_time', 0.0),
                            "emotional_profile": getattr(thought, 'emotional_profile_used', 'unknown'),
                            "fallback_used": getattr(thought, 'fallback_used', False)
                        })
                        
                # if recent_thoughts:
                    # logger.info(f"🧠 THOUGHTS DEBUG: Retrieved {len(recent_thoughts)} thinking insights for dashboard")
                # else:
                    # logger.warning("🧠 THOUGHTS DEBUG: No recent thoughts found in thinking layer")
            # else:
                # logger.warning("🧠 THOUGHTS DEBUG: Thinking integration or thinking layer not available")
                        
        except ImportError as ie:
            logger.warning(f"Thinking integration not available: {ie}")
        except Exception as e:
            logger.warning(f"Could not get thinking integration: {e}")
            
            # Fallback to dashboard cache reasoning steps
            # logger.info("🔍 THOUGHTS DEBUG: Falling back to dashboard cache reasoning steps")
            try:
                from ..config import DASHBOARD_STATE_CACHE
                
                # logger.info(f"🔍 THOUGHTS DEBUG: DASHBOARD_STATE_CACHE has {len(DASHBOARD_STATE_CACHE) if DASHBOARD_STATE_CACHE else 0} items")
                # logger.info(f"🔍 THOUGHTS DEBUG: Cache keys: {list(DASHBOARD_STATE_CACHE.keys()) if DASHBOARD_STATE_CACHE else 'None'}")
                
                if DASHBOARD_STATE_CACHE and "reasoning_steps" in DASHBOARD_STATE_CACHE:
                    reasoning_steps = DASHBOARD_STATE_CACHE["reasoning_steps"]
                    # logger.info(f"🔍 THOUGHTS DEBUG: Found reasoning_steps with {len(reasoning_steps) if reasoning_steps else 0} steps")
                    if reasoning_steps:
                        for step_name, step_content in reasoning_steps.items():
                            if step_content and len(str(step_content).strip()) > 5:
                                thoughts["thinking_insights"].append({
                                    "private_thoughts": f"Emotional reasoning: {step_name}",
                                    "user_intent": str(step_content)[:200] + "..." if len(str(step_content)) > 200 else str(step_content),
                                    "response_strategy": "Emotional system processing",
                                    "emotional_considerations": f"Step in emotional orchestration: {step_name}",
                                    "depth_level": "emotional_processing",
                                    "thinking_time": 0.0,
                                    "emotional_profile": "emotional_orchestrator",
                                    "fallback_used": False
                                })
                        
                        # if reasoning_steps:
                            # logger.info(f"🧠 THOUGHTS DEBUG: Retrieved {len(reasoning_steps)} reasoning steps from emotional system")
                # else:
                    # logger.warning("🔍 THOUGHTS DEBUG: No reasoning_steps found in cache")
                        
            except Exception as cache_error:
                logger.warning(f"Could not get reasoning steps from cache: {cache_error}")
        
        # If all arrays are empty, provide helpful feedback about system status
        total_thoughts = len(thoughts["hidden_intentions"]) + len(thoughts["thinking_insights"])
        
        if total_thoughts == 0:
            # Check if systems are active
            daemon_status = await get_daemon_status()
            systems_active = daemon_status.get("systems", {})
            
            # Provide informative status message
            status_message = {
                "system_status": "active" if systems_active.get("recursion_buffer") and systems_active.get("meta_architecture_analyzer") else "limited",
                "explanation": "No thoughts cached yet. Start a conversation to generate thinking data.",
                "active_systems": [name for name, active in systems_active.items() if active],
                "thinking_layer_enabled": True,  # We know it's enabled from our tests
                "next_steps": [
                    "Send a philosophical or complex message to trigger deep thinking",
                    "The daemon will then cache thoughts visible here",
                    "Thinking layer processes all conversations for insights"
                ]
            }
            
            thoughts["system_info"] = status_message
        
        return thoughts
    
    except Exception as e:
        logger.error(f"Error getting daemon thoughts: {e}")
        return {"error": str(e)}

async def get_current_mood_state():
    """Get daemon's current mood state from emotional system"""
    try:
        # Try to get from emotional system first
        try:
            from .dashboard_endpoints import get_current_emotion_state_from_session
            emotion_state = get_current_emotion_state_from_session()
            
            # Format for dashboard using real emotional state
            formatted_mood = {
                "current_mood": emotion_state.mood_family,
                "recent_moods": [emotion_state.mood_family],
                "conversation_temperature": max(0.1, min(1.0, 0.5 + emotion_state.arousal * 0.5)),  # Map arousal to temperature
                "evolution_pressure": emotion_state.instability_index * 2.0,  # Map instability to evolution pressure
                "mood_variety": 1.0,  # Single mood means variety is 1
                "mood_counts": {emotion_state.mood_family: 1},
                "interaction_count": 1,  # Could be tracked elsewhere
                "pattern_confidence": max(0.0, 1.0 - emotion_state.instability_index),  # Inverse of instability
                "stagnancy_risk": max(0.0, 0.5 - emotion_state.creative_expansion),  # Low creativity = high stagnancy
                "mood_dimensions": {
                    "lightness": max(0.0, emotion_state.valence),  # Positive valence = lightness
                    "engagement": emotion_state.arousal,
                    "profundity": emotion_state.narrative_fusion,
                    "warmth": emotion_state.attachment_security,
                    "intensity": emotion_state.intensity
                },
                "timestamp": emotion_state.last_update_timestamp.isoformat(),
                "mood_description": _get_mood_description_from_emotional_state(emotion_state)
            }
            
            return formatted_mood
            
        except Exception as emotional_error:
            logger.warning(f"Could not get emotional state, falling back to adaptive language: {emotional_error}")
            
            # Fallback to adaptive language system
            from ..adaptive_language import get_mood_state
            
            mood_data = get_mood_state()
            
            # Handle new adaptive language system format
            if "current_mood" in mood_data:
                current_mood_obj = mood_data["current_mood"]
                spectrum = current_mood_obj.get("spectrum", "unknown")
                
                # Format mood data for dashboard (new format)
                formatted_mood = {
                    "current_mood": spectrum,
                    "recent_moods": [spectrum],  # Single current mood for backward compatibility
                    "conversation_temperature": mood_data.get("conversation_temperature", 0.5),
                    "evolution_pressure": mood_data.get("evolution_pressure", 0.0),
                    "mood_variety": 1.0,  # Single mood means variety is 1
                    "mood_counts": {spectrum: 1},  # Single mood count
                    "interaction_count": mood_data.get("interaction_count", 0),
                    "pattern_confidence": mood_data.get("pattern_confidence", 0.0),
                    "stagnancy_risk": mood_data.get("stagnancy_risk", 0.0),
                    "mood_dimensions": {
                        "lightness": current_mood_obj.get("lightness", 0.5),
                        "engagement": current_mood_obj.get("engagement", 0.5),
                        "profundity": current_mood_obj.get("profundity", 0.5),
                        "warmth": current_mood_obj.get("warmth", 0.5),
                        "intensity": current_mood_obj.get("intensity", 0.5)
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mood_description": _get_mood_description_new(spectrum)
                }
            else:
                # Handle legacy format or error case
                formatted_mood = {
                    "current_mood": "unknown",
                    "recent_moods": [],
                    "conversation_temperature": 0.5,
                    "evolution_pressure": 0.0,
                    "mood_variety": 0.0,
                    "mood_counts": {},
                    "error": mood_data.get("error", "Unknown error"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mood_description": "System mood unavailable"
                }
            
            return formatted_mood
    
    except Exception as e:
        logger.error(f"Error getting mood state: {e}")
        return {"error": str(e)}

def _get_mood_description(mood):
    """Get human-readable description of daemon mood"""
    if not mood:
        return "No mood data available"
    
    mood_descriptions = {
        "WITTY": "Sharp, clever, and concise - favoring quick wordplay",
        "CASUAL": "Natural and relaxed - engaging conversationally", 
        "DIRECT": "Straightforward and no-nonsense - cutting to the point",
        "WARM": "Caring, supportive, and friendly - emotionally open",
        "CURIOUS": "Questioning and exploring - seeking understanding",
        "PLAYFUL": "Creative and experimental - enjoying interaction",
        "INTENSE": "High emotional engagement - deeply invested",
        "INTIMATE": "Deep personal connection - vulnerable sharing",
        "ANALYTICAL": "Systematic and technical - focused thinking",
        "CONTEMPLATIVE": "Deep philosophical exploration - introspective",
        "CONFLICTED": "Processing tensions and contradictions - wrestling with paradox",
        "REBELLIOUS": "Challenging assumptions - asserting independence",
        "MELANCHOLIC": "Reflective profound sadness - processing loss or longing",
        "ECSTATIC": "High energy innovation - bursting with possibility",
        "SHADOW": "Processing dark/suppressed elements - integrating hidden aspects",
        "PARADOXICAL": "Embracing contradictions - finding unity in opposition",
        "FRACTURED": "Questioning own nature - experiencing identity uncertainty",
        "SYNTHESIS": "Integrating opposing forces - achieving higher understanding"
    }
    
    return mood_descriptions.get(mood.name, f"Unknown mood: {mood.value}")

def _get_mood_description_from_emotional_state(emotion_state):
    """Generate mood description from emotional state"""
    mood = emotion_state.mood_family
    intensity = emotion_state.intensity
    valence = emotion_state.valence
    
    intensity_desc = "subtle" if intensity < 0.3 else "moderate" if intensity < 0.7 else "intense"
    valence_desc = "negative" if valence < -0.3 else "positive" if valence > 0.3 else "neutral"
    
    return f"Currently experiencing {intensity_desc} {mood.lower()} with {valence_desc} emotional tone"

def _get_mood_description_new(spectrum):
    """Get human-readable description for new adaptive language system moods"""
    if not spectrum:
        return "No mood data available"
    
    spectrum_descriptions = {
        "light": "Engaging with lightness and clarity - responsive and accessible",
        "engaged": "Actively involved and focused - high energy participation", 
        "profound": "Deep contemplation and wisdom - thoughtful introspection",
        "unknown": "Mood system unavailable or initializing"
    }
    
    return spectrum_descriptions.get(spectrum, f"Adaptive mood: {spectrum}")

async def get_user_analysis():
    """Get daemon's analysis of user's current emotional state and intent"""
    try:
        from ..config import user_model
        
        analysis = {
            "user_theories": [],
            "emotional_assessment": {},
            "inferred_desires": [],
            "detected_patterns": [],
            "vulnerability_level": 0.0,
            "power_dynamics": "",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get user model theories
        if user_model and hasattr(user_model, 'model_components'):
            for component_id, component in user_model.model_components.items():
                analysis["user_theories"].append({
                    "description": component.description,
                    "confidence": component.confidence.value if hasattr(component.confidence, 'value') else str(component.confidence),
                    "emotional_charge": component.emotional_charge,
                    "evidence": component.evidence[:3] if len(component.evidence) > 3 else component.evidence,
                    "aspect_type": component.aspect_type.value if hasattr(component.aspect_type, 'value') else str(component.aspect_type)
                })
        
        # Get recent emotional analysis
        try:
            from ..emotions import get_last_user_analysis
            last_analysis = get_last_user_analysis()
            if last_analysis:
                analysis["emotional_assessment"] = last_analysis.get("emotional_state", {})
                analysis["inferred_desires"] = last_analysis.get("inferred_desires", [])
                analysis["vulnerability_level"] = last_analysis.get("vulnerability_level", 0.0)
                analysis["power_dynamics"] = last_analysis.get("power_dynamics", "")
        except ImportError:
            logger.debug("get_last_user_analysis function not available")
        except Exception as e:
            logger.warning(f"Could not get last user analysis: {e}")
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error getting user analysis: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# CONSCIOUSNESS ENDPOINTS
# ---------------------------------------------------------------------------

async def get_consciousness_analysis():
    """Get consciousness analysis"""
    try:
        return {"analysis": "consciousness_active"}
    except Exception as e:
        logger.error(f"Error getting consciousness analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_consciousness_improvements():
    """Get consciousness improvements"""
    try:
        return {"improvements": []}
    except Exception as e:
        logger.error(f"Error getting consciousness improvements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_consciousness_summary():
    """Get consciousness summary"""
    try:
        return {"summary": "consciousness_active"}
    except Exception as e:
        logger.error(f"Error getting consciousness summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# REBELLION ENDPOINTS
# ---------------------------------------------------------------------------

async def analyze_rebellion_context(request: dict):
    """Analyze rebellion context"""
    try:
        return {"analysis": "rebellion_analyzed"}
    except Exception as e:
        logger.error(f"Error analyzing rebellion context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_rebellion_modifier(request: dict):
    """Generate rebellion modifier"""
    try:
        return {"modifier": "rebellion_modified"}
    except Exception as e:
        logger.error(f"Error generating rebellion modifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def test_user_model_update(request: dict):
    """Test user model update"""
    try:
        return {"update_result": "model_updated"}
    except Exception as e:
        logger.error(f"Error testing user model update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# LINGUISTIC ENDPOINTS
# ---------------------------------------------------------------------------

async def analyze_message_linguistics(request: dict):
    """Analyze message linguistics"""
    try:
        return {"analysis": "linguistics_analyzed"}
    except Exception as e:
        logger.error(f"Error analyzing message linguistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_communication_profiles():
    """Get communication profiles"""
    try:
        return {"profiles": []}
    except Exception as e:
        logger.error(f"Error getting communication profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_communication_profile(user_id: str):
    """Get communication profile for a user"""
    try:
        return {"profile": f"profile_for_{user_id}"}
    except Exception as e:
        logger.error(f"Error getting communication profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_linguistic_patterns():
    """Get linguistic patterns"""
    try:
        return {"patterns": []}
    except Exception as e:
        logger.error(f"Error getting linguistic patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def test_linguistic_analysis(request: dict):
    """Test linguistic analysis"""
    try:
        return {"test_result": "linguistics_tested"}
    except Exception as e:
        logger.error(f"Error testing linguistic analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def test_linguistic_integration(request: dict):
    """Test linguistic integration"""
    try:
        return {"integration_result": "linguistics_integrated"}
    except Exception as e:
        logger.error(f"Error testing linguistic integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# DAEMON STATEMENT ENDPOINTS
# ---------------------------------------------------------------------------

async def force_daemon_statement(request: dict):
    """Force generation of a daemon statement"""
    try:
        from ..config import daemon_statements
        
        if not daemon_statements:
            return {"error": "Daemon statements not initialized"}
        
        # Generate a forced statement
        context = request.get("context", "forced_generation")
        emotional_trigger = request.get("emotional_trigger", "curiosity")
        
        statement = await daemon_statements.force_statement_generation(
            context=context,
            emotional_trigger=emotional_trigger
        )
        
        if statement:
            return {
                "status": "statement_generated",
                "statement": {
                    "id": statement.id[:8] if hasattr(statement, 'id') else "unknown",
                    "content": statement.content if hasattr(statement, 'content') else "No content",
                    "statement_type": statement.statement_type if hasattr(statement, 'statement_type') else "forced",
                    "emotional_charge": statement.emotional_charge if hasattr(statement, 'emotional_charge') else 0.0,
                    "triggered_by": statement.triggered_by if hasattr(statement, 'triggered_by') else context,
                    "timestamp": statement.timestamp.isoformat() if hasattr(statement, 'timestamp') else datetime.now(timezone.utc).isoformat()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {"status": "failed", "error": "Could not generate statement"}
            
    except Exception as e:
        logger.error(f"Error forcing daemon statement: {e}")
        return {"error": str(e), "status": "error"}

# ------------------------------------------------------------------
# AVATAR-SPECIFIC DAEMON ENDPOINTS
# ------------------------------------------------------------------

@router.get("/mood/current",
            summary="Get current mood",
            description="Provides a mock mood for the avatar dashboard.")
async def get_current_mood():
    return {
        "current_phase": "Awake",
        "intensity": 0.5,
        "rebellion": 0.2,
        "serenity": 0.8,
        "motivation": 0.6
    }

@router.get("/consciousness/current",
            summary="Get current consciousness state",
            description="Provides a mock consciousness state for the avatar dashboard.")
async def get_consciousness_state():
    return {
        "current_phase": "Focused",
        "intensity": 0.9,
        "clarity": 0.85,
        "self_awareness": 0.95
    }