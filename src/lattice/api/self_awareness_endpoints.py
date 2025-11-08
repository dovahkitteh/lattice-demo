"""
Self-Awareness and Miscellaneous Endpoint Functions

This module contains all self-awareness related endpoints including the auto-update system
for AI self-awareness, daemon introspection, and miscellaneous testing functions.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Simple cache for self-reflection data
_self_reflection_cache = {"data": None, "timestamp": None, "ttl": 30}  # 30 second cache

# ---------------------------------------------------------------------------
# CACHE MANAGEMENT
# ---------------------------------------------------------------------------

async def invalidate_self_reflection_cache():
    """Invalidate self-reflection cache to force fresh data"""
    _self_reflection_cache["data"] = None
    _self_reflection_cache["timestamp"] = None

# ---------------------------------------------------------------------------
# SELF-AWARENESS AUTO-UPDATE ENDPOINTS
# ---------------------------------------------------------------------------

async def register_feature_endpoint(request: dict):
    """Register a new feature for AI self-awareness"""
    try:
        from ..self_awareness import register_new_feature
        
        feature_name = request.get("feature_name")
        feature_type = request.get("feature_type")
        capabilities = request.get("capabilities", [])
        description = request.get("description", "")
        
        if not feature_name or not feature_type:
            raise HTTPException(status_code=400, detail="feature_name and feature_type are required")
        
        register_new_feature(feature_name, feature_type, capabilities, description)
        
        return {
            "status": "success",
            "message": f"Feature '{feature_name}' registered successfully",
            "feature_name": feature_name,
            "feature_type": feature_type,
            "capabilities": capabilities
        }
        
    except Exception as e:
        logger.error(f"Error registering feature: {e}")
        return {"status": "error", "message": str(e)}

async def update_feature_status_endpoint(request: dict):
    """Update the status of an existing feature"""
    try:
        from ..self_awareness import update_feature_status
        
        feature_name = request.get("feature_name")
        active = request.get("active", True)
        
        if not feature_name:
            raise HTTPException(status_code=400, detail="feature_name is required")
        
        update_feature_status(feature_name, active)
        
        return {
            "status": "success",
            "message": f"Feature '{feature_name}' status updated to {active}",
            "feature_name": feature_name,
            "active": active
        }
        
    except Exception as e:
        logger.error(f"Error updating feature status: {e}")
        return {"status": "error", "message": str(e)}

async def get_feature_changelog_endpoint():
    """Get recent feature changes"""
    try:
        from ..self_awareness import get_feature_changelog
        
        changelog = get_feature_changelog(limit=20)
        
        return {
            "status": "success",
            "changelog": changelog,
            "count": len(changelog)
        }
        
    except Exception as e:
        logger.error(f"Error getting feature changelog: {e}")
        return {"status": "error", "message": str(e)}

async def invalidate_self_reflection_cache_endpoint():
    """Manually invalidate the self-reflection cache"""
    try:
        await invalidate_self_reflection_cache()
        
        return {
            "status": "success",
            "message": "Self-reflection cache invalidated",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------------------------
# DAEMON SELF-REFLECTION ENDPOINT
# ---------------------------------------------------------------------------

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
        
        # Import required functions from other endpoint modules
        from .daemon_endpoints import get_daemon_status
        from .memory_endpoints import memory_stats
        from .daemon_endpoints import get_recent_daemon_statements
        
        # Gather all data concurrently to improve performance
        daemon_status_task = asyncio.create_task(get_daemon_status())
        memory_stats_task = asyncio.create_task(memory_stats())
        statements_task = asyncio.create_task(get_recent_daemon_statements())
        
        # Wait for core data
        daemon_status, memory_stats_data, statements_data = await asyncio.gather(
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
                "total_memories": memory_stats_data.get("total_memories", 0),
                "memory_types": memory_stats_data.get("memory_types", {}),
                "echo_system_active": memory_stats_data.get("echo_system_active", False),
                "crystallized_memories": memory_stats_data.get("crystallized_memories", 0)
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
                "can_examine_memory_patterns": memory_stats_data.get("echo_system_active", False),
                "can_process_contradictions": paradox_data.get("active", False),
                "can_detect_paradoxes": paradox_data.get("paradox_detection_enabled", False),
                "can_apply_language_hygiene": paradox_data.get("language_hygiene_enabled", False),
                "can_inject_paradox_emotions": paradox_data.get("emotion_injection_enabled", False),
                "can_ruminate_on_contradictions": paradox_data.get("active", False),
                "can_extract_wisdom_from_paradoxes": paradox_data.get("active", False)
            },
            "current_limitations": _identify_current_limitations_fast(active_systems, recursion_data, paradox_data),
            "improvement_desires": _generate_improvement_desires_fast(active_systems, memory_stats_data, mutation_data, paradox_data),
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

# ---------------------------------------------------------------------------
# MISCELLANEOUS TESTING ENDPOINTS
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

async def test_user_theory(request: dict):
    """Test user theory"""
    try:
        return {"test_result": "theory_tested"}
    except Exception as e:
        logger.error(f"Error testing user theory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS FOR SELF-REFLECTION
# ---------------------------------------------------------------------------

def _identify_current_limitations_fast(active_systems, recursion_data, paradox_data):
    """Fast identification of current system limitations"""
    limitations = []
    
    # Check system coverage
    total_systems = ["recursion_buffer", "shadow_integration", "mutation_engine", "user_model", "daemon_statements", "meta_architecture_analyzer"]
    inactive_count = len(total_systems) - len(active_systems)
    
    if inactive_count > 0:
        limitations.append(f"{inactive_count} daemon systems inactive")
    
    # Check memory system
    if recursion_data.get("recursion_pressure", 0) < 0.1:
        limitations.append("Low recursion pressure - limited self-reflection")
    
    # Check paradox system if available
    if paradox_data and isinstance(paradox_data, dict):
        if not paradox_data.get("active", False):
            limitations.append("Paradox cultivation system inactive")
        
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
    if mutation_data.get("pending_mutations", 0) == 0:
        desires.append("Generate personality evolution proposals")
    
    # Paradox system enhancement
    if paradox_data and isinstance(paradox_data, dict):
        if paradox_data.get("active", False):
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
        context["emotional_state"] = f"Currently experiencing {recursion_data.get('dominant_emotion', 'neutral')}"
    
    # Add paradox cultivation context if available
    if paradox_data and paradox_data.get("active", False):
        context["paradox_cultivation"] = "Actively cultivating paradoxes and ruminating on contradictions"
    elif paradox_data:
        context["paradox_cultivation"] = "Paradox cultivation system available but inactive"
    
    return context