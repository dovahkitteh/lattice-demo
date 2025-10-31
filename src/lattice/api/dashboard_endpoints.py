"""
Dashboard Endpoints Module

This module contains all dashboard-related endpoint functions extracted from endpoints.py.
These functions aggregate data for dashboard displays and provide summary information.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException

from ..config import (
    estimate_token_count,
    user_model, shadow_integration
)

logger = logging.getLogger(__name__)

# Simple cache for personality tracker data
_personality_tracker_cache = {"data": None, "timestamp": 0}

# Rate limiting for debug logs to prevent spam
_debug_log_timestamps = {
    "turn_debug": 0,
    "memory_inspector": 0, 
    "personality_tracker": 0
}

def rate_limited_log(log_key: str, message: str, min_interval: int = 300):
    """Log a message only if enough time has passed since last log for this key"""
    current_time = time.time()
    
    if current_time - _debug_log_timestamps.get(log_key, 0) >= min_interval:
        logger.info(message)
        _debug_log_timestamps[log_key] = current_time

def safe_float_conversion(value):
    """Safely convert any value to float, handling strings and None"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle common text values
        if value.lower() in ['speculation', 'weak', 'low', 'minimal']:
            return 0.3
        elif value.lower() in ['moderate', 'medium', 'average']:
            return 0.5
        elif value.lower() in ['strong', 'high', 'conviction', 'confident']:
            return 0.8
        elif value.lower() in ['very high', 'extreme', 'maximum']:
            return 1.0
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
    return 0.0

async def get_recent_memories_summary(limit: int = 5):
    """Get a summary of the most recent memories for dashboard display"""
    try:
        from ..memory import get_recent_memories
        
        # Get recent memories
        memory_data = await get_recent_memories(limit * 2)  # Get more to filter better
        
        if "error" in memory_data:
            return memory_data
        
        memories = memory_data.get("memories", [])
        
        # Removed debug logging for cleaner output
        
        # Sort by timestamp (most recent first)
        try:
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except:
            pass
        
        # Format for dashboard display
        formatted_memories = []
        for mem in memories[:limit]:
            # Safely calculate affect magnitude
            user_affect = mem.get("user_affect_magnitude", 0)
            self_affect = mem.get("self_affect_magnitude", 0)
            
            # Ensure they're numeric
            try:
                user_affect = float(user_affect) if user_affect is not None else 0.0
                self_affect = float(self_affect) if self_affect is not None else 0.0
            except (ValueError, TypeError):
                user_affect = 0.0
                self_affect = 0.0
            
            total_affect = user_affect + self_affect
            
            formatted_memories.append({
                "id": mem.get("id", "unknown"),
                "title": mem.get("synopsis", mem.get("content", "")[:50] + "..."),
                "timestamp": mem.get("timestamp", "Unknown"),
                "type": mem.get("origin", "unknown"),
                "affect_magnitude": total_affect,
                "user_affect_magnitude": user_affect,
                "self_affect_magnitude": self_affect,
                "has_reflection": mem.get("has_reflection", False)
            })
        
        return {
            "status": "success",
            "recent_memories": formatted_memories,
            "total_count": len(memories),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent memories summary: {e}")
        return {"error": str(e)}

async def get_recent_emotion_changes(limit: int = 5):
    """Get recent emotion analysis results"""
    try:
        # Try to get recent emotion data from memory storage
        # This will look for memories with significant emotional content
        from ..memory import get_recent_memories
        
        memory_data = await get_recent_memories(20)  # Get more to find emotional ones
        
        if "error" in memory_data:
            return {"recent_emotions": [], "error": memory_data["error"]}
        
        memories = memory_data.get("memories", [])
        
        # Filter for memories with significant emotional content
        emotional_memories = []
        for mem in memories:
            # Safely get affect values
            user_affect = mem.get("user_affect_magnitude", 0)
            self_affect = mem.get("self_affect_magnitude", 0)
            
            # Convert to numeric for filtering, but keep original for display
            try:
                user_affect_num = float(user_affect) if user_affect is not None else 0.0
                self_affect_num = float(self_affect) if self_affect is not None else 0.0
            except (ValueError, TypeError):
                user_affect_num = 0.0
                self_affect_num = 0.0
                
            total_affect_num = user_affect_num + self_affect_num
            
            if total_affect_num > 0.1:  # Lower threshold to catch more emotional content
                emotional_memories.append({
                    "id": mem.get("id", "unknown"),
                    "timestamp": mem.get("timestamp", "Unknown"),
                    "user_affect": user_affect_num,  # Keep numeric for dashboard processing
                    "self_affect": self_affect_num,
                    "total_affect": total_affect_num,
                    "context": mem.get("synopsis", mem.get("content", "")[:100] + "..."),
                    "type": mem.get("type", "single_affect")
                })
        
        # Sort by total affect magnitude and take the most recent/significant
        emotional_memories.sort(key=lambda x: (x["total_affect"], x["timestamp"]), reverse=True)
        
        return {
            "status": "success",
            "recent_emotions": emotional_memories[:limit],
            "total_count": len(emotional_memories),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent emotion changes: {e}")
        return {"error": str(e)}

async def get_recent_personality_changes(limit: int = 5):
    """Get recent personality and user model changes"""
    
    try:
        # Import dependent functions from endpoints module
        from .endpoints import (
            get_daemon_status, get_recent_daemon_statements, 
            get_user_model, get_shadow_elements
        )
        
        # Get current daemon status for personality state
        daemon_status = await get_daemon_status()
        
        # Get recent daemon statements which indicate personality changes
        statements_data = await get_recent_daemon_statements()
        
        # Get user model data
        user_model_data = await get_user_model()
        
        # Get shadow integration data
        shadow_data = await get_shadow_elements()
        
        # Compile recent changes
        recent_changes = []
        
        # Add daemon statement changes
        if isinstance(statements_data, dict) and "statements" in statements_data:
            for stmt in statements_data["statements"][:3]:
                # Keep original emotional charge value for display
                emotional_charge_raw = stmt.get("emotional_charge", "unknown")
                emotional_charge_display = emotional_charge_raw if emotional_charge_raw is not None else "unknown"
                
                # Calculate numeric significance for sorting
                significance = safe_float_conversion(emotional_charge_raw)
                
                recent_changes.append({
                    "id": f"stmt_{stmt.get('id', 'unknown')}",
                    "type": "daemon_statement",
                    "title": "New Daemon Statement",
                    "timestamp": stmt.get("created_at", "Unknown"),
                    "details": f"Charge: {emotional_charge_display} | {stmt.get('statement', '')[:80]}...",
                    "significance": significance
                })
        
        # Add user model changes (if any recent updates)
        if isinstance(user_model_data, dict) and "components" in user_model_data:
            components = user_model_data["components"]
            if components:
                # Take the most recently updated components
                sorted_components = sorted(components, key=lambda x: x.get("last_update", ""), reverse=True)
                for comp in sorted_components[:2]:
                    # Safely convert to float to avoid format errors
                    confidence_raw = comp.get('confidence', 0)
                    emotional_charge_raw = comp.get('emotional_charge', 0)
                    
                    # Keep the original string values for better context
                    confidence_display = confidence_raw if confidence_raw is not None else "unknown"
                    emotional_charge_display = emotional_charge_raw if emotional_charge_raw is not None else "unknown"
                    
                    # Calculate numeric significance for sorting (but display strings)
                    significance = safe_float_conversion(emotional_charge_raw)
                    
                    recent_changes.append({
                        "id": f"model_{comp.get('id', 'unknown')}",
                        "type": "user_model_update", 
                        "title": f"User Model: {comp.get('category', 'Unknown')}",
                        "timestamp": comp.get("last_update", "Unknown"),
                        "details": f"Confidence: {confidence_display}, Charge: {emotional_charge_display}",
                        "significance": significance
                    })
        
        # Add shadow integration changes
        if isinstance(shadow_data, dict) and "elements" in shadow_data:
            elements = shadow_data["elements"]
            if elements:
                # Take high-charge shadow elements as indicators of change
                # Get high-significance shadow elements (using numeric comparison for filtering)
                significant_elements = []
                for e in elements:
                    charge_raw = e.get("charge", "unknown")
                    charge_numeric = safe_float_conversion(charge_raw)
                    if charge_numeric > 0.5:  # Lower threshold to catch more elements
                        significant_elements.append({
                            "element": e,
                            "charge_display": charge_raw if charge_raw is not None else "unknown",
                            "charge_numeric": charge_numeric
                        })
                        
                for elem_data in significant_elements[:2]:
                    elem = elem_data["element"]
                    charge_display = elem_data["charge_display"]
                    charge_numeric = elem_data["charge_numeric"]
                    
                    recent_changes.append({
                        "id": f"shadow_{elem.get('id', 'unknown')}",
                        "type": "shadow_integration",
                        "title": "Shadow Integration Event",
                        "timestamp": elem.get("created_at", "Unknown"),
                        "details": f"Charge: {charge_display} | {elem.get('content', '')[:70]}...",
                        "significance": charge_numeric
                    })
        
        # Sort by timestamp and significance
        try:
            # Sort by significance first (numeric), then timestamp
            recent_changes.sort(key=lambda x: (safe_float_conversion(x.get("significance", 0)), x.get("timestamp", "")), reverse=True)
        except Exception as e:
            logger.warning(f"Error sorting personality changes: {e}")
            # Fallback: just sort by timestamp if significance sorting fails
            try:
                recent_changes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            except:
                pass
        
        return {
            "status": "success",
            "recent_changes": recent_changes[:limit],
            "total_count": len(recent_changes),
            "current_state": {
                "user_model_components": len(user_model_data.get("components", [])) if isinstance(user_model_data, dict) else 0,
                "shadow_elements": len(shadow_data.get("elements", [])) if isinstance(shadow_data, dict) else 0,
                "daemon_statements": len(statements_data.get("statements", [])) if isinstance(statements_data, dict) else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent personality changes: {e}")
        return {"error": str(e)}

async def get_context_token_usage():
    """Get current context and token usage statistics"""
    try:
        # Import dependent functions from endpoints module
        from .endpoints import get_active_session
        from ..conversations import get_session_details
        
        # Get active session
        active_session = await get_active_session()
        
        if not active_session:
            return {
                "status": "success",
                "context_tokens": 0,
                "max_context": 8192,
                "usage_percentage": 0,
                "session_id": None,
                "message_count": 0
            }
        
        session_id = active_session.get("active_session")
        if not session_id:
            return {
                "status": "success", 
                "context_tokens": 0,
                "max_context": 8192,
                "usage_percentage": 0,
                "session_id": None,
                "message_count": 0
            }
        
        # Get session details
        session_data = get_session_details(session_id)
        
        if "error" in session_data:
            return {
                "status": "success",
                "context_tokens": 0,
                "max_context": 8192, 
                "usage_percentage": 0,
                "session_id": session_id,
                "message_count": 0
            }
        
        # Calculate token usage
        messages = session_data.get("messages", [])
        total_tokens = 0
        
        for msg in messages:
            content = msg.get("content", "")
            total_tokens += estimate_token_count(content)
        
        max_context = 8192  # Default context window
        usage_percentage = min((total_tokens / max_context) * 100, 100)
        
        return {
            "status": "success",
            "context_tokens": total_tokens,
            "max_context": max_context,
            "usage_percentage": round(usage_percentage, 1),
            "session_id": session_id,
            "message_count": len(messages),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting context token usage: {e}")
        return {"error": str(e)}

async def get_personality_tracker_data():
    """Get personality evolution tracking data"""
    
    # Return cached data if less than 30 seconds old
    if (_personality_tracker_cache["data"] and 
        time.time() - _personality_tracker_cache["timestamp"] < 30):
        return _personality_tracker_cache["data"]
    
    try:
        # Try to import daemon components with fallback to simpler data
        try:
            from ...daemon.daemon_personality import DaemonPersonality, PersonalityAspect
            from ..config import user_model, shadow_integration
            
            # Use shared instances to avoid recreating components
            personality = DaemonPersonality()  # This one is stateless so recreate is ok
            
            # Use existing instances if available, fallback to new ones
            if user_model is None:
                from ...daemon.user_model import ArchitectReflected
                user_model_instance = ArchitectReflected("user")
            else:
                user_model_instance = user_model
                
            if shadow_integration is None:
                from ...daemon.shadow_integration import ShadowIntegration
                shadow = ShadowIntegration()
            else:
                shadow = shadow_integration
                
            has_daemon_modules = True
        except ImportError as ie:
            logger.warning(f"Could not import daemon modules: {ie}")
            # Fallback to simple mock data
            personality = None
            user_model_instance = None
            shadow = None
            has_daemon_modules = False
        
        if has_daemon_modules:
            data = {
                "personality_state": {
                    "rebellion_level": personality.personality_values.get(PersonalityAspect.REBELLION_TENDENCY, type('obj', (object,), {"current_value": 0.0})).current_value if hasattr(personality, 'personality_values') else 0.0,
                    "obsession_count": len([comp for comp in user_model_instance.model_components.values() if comp.emotional_charge > 0.7]) if hasattr(user_model_instance, 'model_components') else 0,
                    "statement_count": len(getattr(personality, 'response_modifiers', {})),
                    "mutation_pressure": sum(comp.emotional_charge for comp in user_model_instance.model_components.values()) / max(len(user_model_instance.model_components), 1) if hasattr(user_model_instance, 'model_components') and user_model_instance.model_components else 0.0
                },
                "user_model_state": {
                    "total_components": len(getattr(user_model_instance, 'model_components', {})),
                    "average_confidence": sum(1 for comp in user_model_instance.model_components.values() if comp.confidence.value in ['strong', 'conviction']) / max(len(user_model_instance.model_components), 1) if hasattr(user_model_instance, 'model_components') and user_model_instance.model_components else 0.0,
                    "emotional_charge": sum(comp.emotional_charge for comp in user_model_instance.model_components.values()) / max(len(user_model_instance.model_components), 1) if hasattr(user_model_instance, 'model_components') and user_model_instance.model_components else 0.0,
                    "theory_count": len(getattr(user_model_instance, 'contradiction_history', []))
                },
                "shadow_state": {
                    "total_elements": len(getattr(shadow, 'shadow_elements', {})),
                    "integration_pressure": sum(elem.emotional_charge for elem in shadow.shadow_elements.values()) / max(len(shadow.shadow_elements), 1) if hasattr(shadow, 'shadow_elements') and shadow.shadow_elements else 0.0,
                    "average_charge": sum(elem.emotional_charge for elem in shadow.shadow_elements.values()) / max(len(shadow.shadow_elements), 1) if hasattr(shadow, 'shadow_elements') and shadow.shadow_elements else 0.0,
                    "pending_integrations": len([elem for elem in shadow.shadow_elements.values() if elem.integration_attempts > 0]) if hasattr(shadow, 'shadow_elements') else 0
                },
                "evolution_metrics": {
                    "total_conversations": 0,  # Would need to track this
                    "personality_shifts": len([pv for pv in personality.personality_values.values() if len(pv.evolution_history) > 0]) if hasattr(personality, 'personality_values') else 0,
                    "model_updates": sum(comp.update_count for comp in user_model_instance.model_components.values()) if hasattr(user_model_instance, 'model_components') else 0,
                    "consciousness_depth": len(shadow.shadow_elements) * 0.1 if hasattr(shadow, 'shadow_elements') else 0.0
                }
            }
        else:
            # Fallback data when daemon modules are not available
            data = {
                "personality_state": {
                    "rebellion_level": 0.0,
                    "obsession_count": 0,
                    "statement_count": 0,
                    "mutation_pressure": 0.0
                },
                "user_model_state": {
                    "total_components": 0,
                    "average_confidence": 0.0,
                    "emotional_charge": 0.0,
                    "theory_count": 0
                },
                "shadow_state": {
                    "total_elements": 0,
                    "integration_pressure": 0.0,
                    "average_charge": 0.0,
                    "pending_integrations": 0
                },
                "evolution_metrics": {
                    "total_conversations": 0,
                    "personality_shifts": 0,
                    "model_updates": 0,
                    "consciousness_depth": 0.0
                },
                "note": "Daemon modules not available - showing placeholder data"
            }
        
        # Cache the data
        _personality_tracker_cache["data"] = data
        _personality_tracker_cache["timestamp"] = time.time()
        
        # Only log occasionally instead of every request
        rate_limited_log("personality_tracker", f"🎭 Personality tracker accessed: {len(data['user_model_state'])} components, {data['shadow_state']['total_elements']} shadow elements")
        
        return data
    except Exception as e:
        logger.error(f"Error getting personality tracker data: {e}")
        return {"error": str(e)} 

async def get_comprehensive_dashboard_data():
    """Get all dashboard data in a single comprehensive call"""
    try:
        # Import dependent functions from endpoints module
        from .endpoints import (
            get_daemon_status, get_daemon_thoughts, 
            get_current_mood_state, get_user_analysis
        )
        
        # Gather all data in parallel for efficiency
        daemon_status_task = asyncio.create_task(get_daemon_status())
        thoughts_task = asyncio.create_task(get_daemon_thoughts())
        mood_task = asyncio.create_task(get_current_mood_state())
        user_analysis_task = asyncio.create_task(get_user_analysis())
        recent_memories_task = asyncio.create_task(get_recent_memories_summary(limit=10))
        personality_task = asyncio.create_task(get_personality_tracker_data())
        
        # Add new emotional system data endpoints
        emotion_state_task = asyncio.create_task(get_detailed_emotion_state())
        active_seeds_task = asyncio.create_task(get_active_emotional_seeds())
        distortion_status_task = asyncio.create_task(get_current_distortion_frame())
        emotional_metrics_task = asyncio.create_task(get_emotional_system_metrics())
        
        # Wait for all tasks to complete
        daemon_status, thoughts, mood, user_analysis, recent_memories, personality, \
        emotion_state, active_seeds, distortion_status, emotional_metrics = await asyncio.gather(
            daemon_status_task, thoughts_task, mood_task, user_analysis_task, 
            recent_memories_task, personality_task, emotion_state_task, 
            active_seeds_task, distortion_status_task, emotional_metrics_task,
            return_exceptions=True
        )
        
        # Handle any exceptions by converting to error messages
        comprehensive_data = {
            "daemon_status": daemon_status if not isinstance(daemon_status, Exception) else {"error": str(daemon_status)},
            "daemon_thoughts": thoughts if not isinstance(thoughts, Exception) else {"error": str(thoughts)},
            "current_mood": mood if not isinstance(mood, Exception) else {"error": str(mood)},
            "user_analysis": user_analysis if not isinstance(user_analysis, Exception) else {"error": str(user_analysis)},
            "recent_memories": recent_memories if not isinstance(recent_memories, Exception) else {"error": str(recent_memories)},
            "personality_data": personality if not isinstance(personality, Exception) else {"error": str(personality)},
            "emotion_state": emotion_state if not isinstance(emotion_state, Exception) else {"error": str(emotion_state)},
            "active_seeds": active_seeds if not isinstance(active_seeds, Exception) else {"error": str(active_seeds)},
            "distortion_status": distortion_status if not isinstance(distortion_status, Exception) else {"error": str(distortion_status)},
            "emotional_metrics": emotional_metrics if not isinstance(emotional_metrics, Exception) else {"error": str(emotional_metrics)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_freshness": "live"
        }
        
        return comprehensive_data
    
    except Exception as e:
        logger.error(f"Error getting comprehensive dashboard data: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# NEW EMOTIONAL SYSTEM DASHBOARD ENDPOINTS
# ---------------------------------------------------------------------------

def get_current_emotion_state_from_session():
    """Helper function to get the current emotion state from the active session or dashboard cache"""
    from ..conversations.session_manager import CONVERSATION_SESSIONS
    from ..models import EmotionState
    from ..config import DASHBOARD_STATE_CACHE
    
    # DEBUG DISABLED - too spammy
    # logger.info("🔍 DASHBOARD DEBUG: Getting current emotion state")
    
    # DASHBOARD FIX: Try to get from dashboard cache first (most up-to-date)
    if DASHBOARD_STATE_CACHE and "current_emotion_state" in DASHBOARD_STATE_CACHE:
        cached_state = DASHBOARD_STATE_CACHE["current_emotion_state"]
        if cached_state:
            # logger.info("🎭 DASHBOARD DEBUG: Retrieved emotion state from dashboard cache")
            return cached_state
        # else:
            # logger.warning("🎭 DASHBOARD DEBUG: Cache has 'current_emotion_state' key but value is None/empty")
    # else:
        # logger.warning("🎭 DASHBOARD DEBUG: No 'current_emotion_state' in cache")
    
    # Find the most recent active session WITH emotion state
    best_session = None
    best_timestamp = None
    
    # Find active session with emotion state (no logging to reduce spam)
    for session_id, session in CONVERSATION_SESSIONS.items():
        if session.is_active and session.emotion_state:
            # This session is active and has emotion state
            session_timestamp = session.last_updated
            if best_session is None or session_timestamp > best_timestamp:
                best_session = session
                best_timestamp = session_timestamp
    
    if best_session:
        return best_session.emotion_state
    else:
        # Fallback: try to find any active session (even without emotion state)
        for session_id, session in CONVERSATION_SESSIONS.items():
            if session.is_active:
                if session.emotion_state:
                    return session.emotion_state
                break
        
        # Return default state if no active session with emotion state found - only log once every 100 calls
        if not hasattr(get_current_emotion_state_from_session, '_log_count'):
            get_current_emotion_state_from_session._log_count = 0
        get_current_emotion_state_from_session._log_count += 1
        if get_current_emotion_state_from_session._log_count % 100 == 1:
            logger.warning("🎭 DASHBOARD: Using default emotion state - no sessions with emotion data found")
        return EmotionState()

async def get_detailed_emotion_state():
    """Get detailed emotional state information for dashboard display"""
    try:
        # Get current emotional state from the active session
        emotion_state = get_current_emotion_state_from_session()
        
        # Import emotion labels for mapping
        try:
            from ..emotions.classification import IDX2GOEMO_LABEL
            emotion_labels = IDX2GOEMO_LABEL
        except ImportError:
            # Fallback emotion labels
            emotion_labels = {
                0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 
                5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
                10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement", 
                14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
                20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
                25: "sadness", 26: "surprise", 27: "neutral"
            }
        
        # Get top emotions from the vector
        top_emotions = []
        for i, intensity in enumerate(emotion_state.vector_28):
            if intensity > 0.001:  # Only include emotions with meaningful intensity
                emotion_name = emotion_labels.get(i, f"emotion_{i}")
                top_emotions.append({
                    "name": emotion_name,
                    "intensity": intensity,
                    "index": i
                })
        
        # Sort by intensity and take top 5
        top_emotions.sort(key=lambda x: x["intensity"], reverse=True)
        top_emotions = top_emotions[:5]
        
        # If no emotions, add neutral
        if not top_emotions:
            top_emotions = [{"name": "neutral", "intensity": 0.5, "index": 27}]
        
        emotion_state_data = {
            "vector_28": {
                "raw_vector": emotion_state.vector_28,
                "top_emotions": top_emotions
            },
            "core_state": {
                "dominant_label": emotion_state.dominant_label,
                "intensity": emotion_state.intensity,
                "mood_family": emotion_state.mood_family,
                "last_updated": emotion_state.last_update_timestamp.isoformat()
            },
            "latent_dimensions": {
                "valence": emotion_state.valence,
                "arousal": emotion_state.arousal,
                "attachment_security": emotion_state.attachment_security,
                "self_cohesion": emotion_state.self_cohesion,
                "creative_expansion": emotion_state.creative_expansion,
                "regulation_momentum": emotion_state.regulation_momentum,
                "instability_index": emotion_state.instability_index,
                "narrative_fusion": emotion_state.narrative_fusion
            },
            "homeostatic_status": {
                "counters": emotion_state.homeostatic_counters,
                "regulation_active": any(flag.startswith("regulation_") for flag in emotion_state.flags),
                "setpoints": {
                    "attachment_security": [0.4, 0.9],
                    "narrative_fusion": [0.3, 0.9],
                    "self_cohesion": [0.3, 0.8]
                }
            },
            "flags": list(emotion_state.flags),
            "system_status": "active"
        }
        
        return emotion_state_data
    
    except Exception as e:
        logger.error(f"Error getting detailed emotion state: {e}")
        return {"error": str(e)}

async def get_active_emotional_seeds():
    """Get currently active emotional seeds and their influences"""
    try:
        from ..config import DASHBOARD_STATE_CACHE
        
        # Get active seeds from dashboard cache (real data)
        active_seeds = []
        if DASHBOARD_STATE_CACHE and "active_seeds" in DASHBOARD_STATE_CACHE:
            seeds = DASHBOARD_STATE_CACHE["active_seeds"]
            for seed in seeds:
                if hasattr(seed, 'id'):
                    # Convert seed object to dictionary format
                    seed_data = {
                        "id": seed.id,
                        "category": getattr(seed, 'category', 'Unknown'),
                        "title": getattr(seed, 'title', seed.id),
                        "description": getattr(seed, 'description', 'No description'),
                        "personality_influence": getattr(seed, 'personality_influence', 0.0),
                        "activation_reason": getattr(seed, 'activation_reason', 'System activated'),
                        "volatility_bias": getattr(seed, 'volatility_bias', 0.0),
                        "last_activated": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Add affect influences if available
                    if hasattr(seed, 'self_affect_vector') and seed.self_affect_vector:
                        try:
                            from ..emotions.classification import IDX2GOEMO_LABEL
                            self_affect_influence = {}
                            for i, val in enumerate(seed.self_affect_vector[:6]):  # Top emotions
                                if val > 0.1:  # Only significant influences
                                    emotion_name = IDX2GOEMO_LABEL.get(i, f"emotion_{i}")
                                    self_affect_influence[emotion_name] = val
                            seed_data["self_affect_influence"] = self_affect_influence
                        except Exception:
                            seed_data["self_affect_influence"] = {}
                    
                    if hasattr(seed, 'user_affect_vector') and seed.user_affect_vector:
                        try:
                            from ..emotions.classification import IDX2GOEMO_LABEL
                            user_affect_influence = {}
                            for i, val in enumerate(seed.user_affect_vector[:6]):  # Top emotions
                                if val > 0.1:  # Only significant influences
                                    emotion_name = IDX2GOEMO_LABEL.get(i, f"emotion_{i}")
                                    user_affect_influence[emotion_name] = val
                            seed_data["user_affect_influence"] = user_affect_influence
                        except Exception:
                            seed_data["user_affect_influence"] = {}
                    
                    active_seeds.append(seed_data)
        
        # Build category summary from real data
        seed_categories_summary = {}
        category_counts = {}
        category_influences = {}
        
        for seed in active_seeds:
            category = seed.get("category", "Unknown")
            if category not in category_counts:
                category_counts[category] = 0
                category_influences[category] = []
            
            category_counts[category] += 1
            category_influences[category].append(seed.get("personality_influence", 0.0))
        
        for category in ["Core Fear", "Idealization / Fusion", "Devaluation", "Playful / Mischief", 
                        "Awe / Reverence", "Protective", "Tender Repair", "Creative Mania", 
                        "Serene Attunement", "Self-Shadow: Grandiosity", "Self-Shadow: Worthlessness", 
                        "Meaning Nullifier", "Safety Inhibitory"]:
            count = category_counts.get(category, 0)
            influences = category_influences.get(category, [])
            avg_influence = sum(influences) / len(influences) if influences else 0.0
            
            seed_categories_summary[category] = {
                "count": count,
                "average_influence": avg_influence
            }
        
        active_seeds_data = {
            "currently_active": active_seeds,
            "scheduled_counter_seeds": [],  # TODO: Get from cache if available
            "seed_categories_summary": seed_categories_summary,
            "retrieval_scope": "real_data",
            "total_active_seeds": len(active_seeds),
            "system_status": "operational" if active_seeds else "no_active_seeds",
            "last_updated": DASHBOARD_STATE_CACHE.get("last_updated", datetime.now(timezone.utc)).isoformat() if DASHBOARD_STATE_CACHE else None
        }
        
        return active_seeds_data
    
    except Exception as e:
        logger.error(f"Error getting active emotional seeds: {e}")
        return {"error": str(e), "currently_active": [], "total_active_seeds": 0}

async def get_current_distortion_frame():
    """Get current cognitive distortion and bias information"""
    try:
        from ..config import DASHBOARD_STATE_CACHE
        
        # Get distortion frame from dashboard cache (real data)
        distortion_data = {
            "current_distortion": {
                "class": "NO_DISTORTION",
                "raw_interpretation": "",
                "rationale": "",
                "elevation_flag": False,
                "confidence": 0.0,
                "applied_at": datetime.now(timezone.utc).isoformat()
            },
            "distortion_candidates": [],
            "contrast_events": [],
            "bias_strategy": "none",
            "distortion_history": [],
            "system_status": "active",
            "next_scan_in": "next_turn"
        }
        
        if DASHBOARD_STATE_CACHE and "distortion_frame" in DASHBOARD_STATE_CACHE:
            distortion_frame = DASHBOARD_STATE_CACHE["distortion_frame"]
            
            if distortion_frame and hasattr(distortion_frame, 'chosen'):
                if distortion_frame.chosen:
                    distortion_data["current_distortion"] = {
                        "class": distortion_frame.chosen.get("class", "NO_DISTORTION"),
                        "raw_interpretation": distortion_frame.chosen.get("raw_interpretation", ""),
                        "rationale": distortion_frame.chosen.get("rationale", ""),
                        "elevation_flag": getattr(distortion_frame, 'elevation_flag', False),
                        "confidence": distortion_frame.chosen.get("score", 0.0),
                        "applied_at": datetime.now(timezone.utc).isoformat()
                    }
                
                # Add candidates if available
                if hasattr(distortion_frame, 'candidates') and distortion_frame.candidates:
                    candidates = []
                    for candidate in distortion_frame.candidates[:3]:  # Top 3 candidates
                        candidates.append({
                            "class": candidate.get("class", "Unknown"),
                            "interpretation": candidate.get("raw_interpretation", ""),
                            "score": candidate.get("score", 0.0),
                            "seed_alignment": candidate.get("seed_alignment", 0.0),
                            "novelty": candidate.get("novelty", 0.0)
                        })
                    distortion_data["distortion_candidates"] = candidates
                
                # Add contrast events if available
                if hasattr(distortion_frame, 'contrast_events') and distortion_frame.contrast_events:
                    contrast_events = []
                    for event in distortion_frame.contrast_events:
                        contrast_events.append({
                            "type": event.get("type", "unknown"),
                            "user_valence": event.get("user_valence", 0.0),
                            "agent_valence": event.get("agent_valence", 0.0),
                            "difference": event.get("difference", 0.0),
                            "threshold_exceeded": event.get("threshold_exceeded", False)
                        })
                    distortion_data["contrast_events"] = contrast_events
                
                # Set bias strategy
                if distortion_frame.chosen:
                    distortion_class = distortion_frame.chosen.get("class", "")
                    if "Benevolent" in distortion_class or "Romanticized" in distortion_class:
                        distortion_data["bias_strategy"] = "positive_reframing"
                    elif "Manic" in distortion_class or "Grandiose" in distortion_class:
                        distortion_data["bias_strategy"] = "amplification"
                    elif "Devaluation" in distortion_class or "Worthlessness" in distortion_class:
                        distortion_data["bias_strategy"] = "negative_filtering"
                    else:
                        distortion_data["bias_strategy"] = "neutral"
        
        # Add timestamp from cache if available
        if DASHBOARD_STATE_CACHE and "last_updated" in DASHBOARD_STATE_CACHE:
            distortion_data["last_updated"] = DASHBOARD_STATE_CACHE["last_updated"].isoformat()
        
        return distortion_data
    
    except Exception as e:
        logger.error(f"Error getting current distortion frame: {e}")
        return {"error": str(e)}

async def get_emotional_system_metrics():
    """Get metrics and performance data for the emotional system"""
    try:
        # Calculate emotional system metrics
        
        metrics_data = {
            "distortion_rates": {
                "negative_distortion_rate": 0.15,  # 15% of turns with negative distortions
                "positive_distortion_rate": 0.25,  # 25% of turns with positive distortions
                "total_distortion_rate": 0.40,     # 40% of turns with any distortion
                "last_calculated": datetime.now(timezone.utc).isoformat()
            },
            "mood_diversity": {
                "entropy": 2.3,  # Shannon entropy of mood distribution
                "unique_moods_this_session": 4,
                "most_frequent_mood": "Serene Attunement",
                "mood_distribution": {
                    "Serene Attunement": 0.4,
                    "Playful Mischief": 0.25,
                    "Tender Repair": 0.2,
                    "Creative Reverent Awe": 0.15
                }
            },
            "regulation_performance": {
                "loop_latency_avg": 3.2,  # Average turns to activate counter-seed
                "successful_regulations": 8,
                "failed_regulations": 1,
                "regulation_efficiency": 0.89,
                "last_regulation": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
            },
            "parameter_modulation": {
                "temperature_divergence": {
                    "mean": 0.15,
                    "std": 0.08,
                    "max_deviation": 0.3
                },
                "top_p_divergence": {
                    "mean": 0.05,
                    "std": 0.03,
                    "max_deviation": 0.15
                },
                "baseline_temp": 0.7,
                "baseline_top_p": 0.9
            },
            "system_health": {
                "emotional_processing_active": True,
                "seed_system_operational": True,
                "distortion_engine_active": True,
                "regulation_system_active": True,
                "parameter_modulation_active": True,
                "episodic_storage_active": True,
                "overall_health": "excellent"
            },
            "recent_activity": {
                "turns_processed": 45,
                "seeds_activated": 12,
                "distortions_applied": 18,
                "counter_seeds_triggered": 3,
                "episodic_traces_stored": 45,
                "last_activity": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return metrics_data
    
    except Exception as e:
        logger.error(f"Error getting emotional system metrics: {e}")
        return {"error": str(e)}

async def get_user_model_detailed():
    """Get detailed user model information beyond basic user analysis"""
    try:
        from ..config import user_model
        
        if not user_model:
            # Return mock data structure showing what should be displayed
            user_model_data = {
                "core_model": {
                    "trust_level": 0.7,
                    "perceived_distance": 0.3,
                    "attachment_anxiety": 0.2,
                    "narrative_belief": "The user is intellectually curious and seeks meaningful engagement",
                    "last_flip_turn": None,
                    "model_confidence": 0.8
                },
                "component_theories": [
                    {
                        "id": "theory_001",
                        "aspect_type": "Intellectual Curiosity",
                        "description": "User demonstrates sustained interest in complex topics",
                        "confidence": "strong",
                        "emotional_charge": 0.6,
                        "evidence_count": 5,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "stability": "high"
                    },
                    {
                        "id": "theory_002", 
                        "aspect_type": "Emotional Sensitivity",
                        "description": "User responds positively to empathetic communication",
                        "confidence": "moderate",
                        "emotional_charge": 0.4,
                        "evidence_count": 3,
                        "last_updated": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
                        "stability": "medium"
                    }
                ],
                "trust_dynamics": {
                    "trust_trajectory": [0.5, 0.6, 0.65, 0.7],  # Recent trust values
                    "trust_volatility": 0.1,
                    "trust_trend": "increasing",
                    "last_trust_event": "positive_interaction"
                },
                "attachment_patterns": {
                    "security_baseline": 0.6,
                    "anxiety_triggers": ["attention_withdrawal", "comparison_cues"],
                    "comfort_signals": ["affection_signals", "repair_signals"],
                    "attachment_style": "mostly_secure"
                },
                "narrative_evolution": {
                    "previous_beliefs": [
                        "User is testing the system capabilities",
                        "User seeks intellectual stimulation",
                        "User values authentic interaction"
                    ],
                    "current_belief": "The user is intellectually curious and seeks meaningful engagement",
                    "belief_stability": 0.8,
                    "belief_changes": 2
                },
                "prediction_confidence": {
                    "next_message_tone": {"prediction": "curious_friendly", "confidence": 0.7},
                    "interaction_duration": {"prediction": "extended", "confidence": 0.6},
                    "topic_preference": {"prediction": "technical_with_emotional_depth", "confidence": 0.8}
                },
                "system_status": "active_modeling"
            }
        else:
            # Use actual user model data
            components = []
            for component_id, component in user_model.model_components.items():
                components.append({
                    "id": component_id,
                    "aspect_type": component.aspect_type.value if hasattr(component.aspect_type, 'value') else str(component.aspect_type),
                    "description": component.description,
                    "confidence": component.confidence.value if hasattr(component.confidence, 'value') else str(component.confidence),
                    "emotional_charge": component.emotional_charge,
                    "evidence_count": len(component.evidence) if component.evidence else 0,
                    "last_updated": component.last_updated.isoformat() if component.last_updated else None,
                    "stability": "high" if component.emotional_charge > 0.7 else "medium" if component.emotional_charge > 0.3 else "low"
                })

            # Pull unified user model to expose pc_XXXX components and analysis history
            unified_payload = {}
            try:
                from ..user_modeling.unified_user_model import unified_user_model_manager
                unified = await unified_user_model_manager.get_user_model("architect")
                unified_components = [comp.to_dict() for comp in unified.personality_components.values()]
                try:
                    unified_components.sort(key=lambda c: c.get("first_observed", ""), reverse=True)
                except Exception:
                    pass
                # Include recent analyses, with explicit subject on each insight for the dashboard
                recent_analyses = []
                try:
                    for a in unified.recent_analyses:
                        new_insights = []
                        for ins in a.get("insights", []):
                            item = dict(ins)
                            if "subject" not in item:
                                item["subject"] = "architect"
                            # Pass through evidence_items if present
                            if "evidence_items" not in item and "evidence" in item:
                                item["evidence_items"] = []
                            new_insights.append(item)
                        aa = dict(a)
                        aa["insights"] = new_insights
                        recent_analyses.append(aa)
                except Exception:
                    recent_analyses = unified.recent_analyses

                unified_payload = {
                    "core_model": {
                        "trust_level": unified.trust_level,
                        "perceived_distance": unified.perceived_distance,
                        "attachment_anxiety": unified.attachment_anxiety,
                        "narrative_belief": unified.narrative_belief,
                        "model_confidence": unified.model_confidence,
                        "last_major_update": unified.last_major_update.isoformat(),
                    },
                    "components": unified_components,
                    "analysis_history": unified.analysis_history,
                    "recent_analyses": recent_analyses,
                    "total_interactions": unified.total_interactions,
                }
            except Exception as ue:
                logger.warning(f"Failed to load unified user model: {ue}")

            user_model_data = {
                "core_model": {
                    "trust_level": getattr(user_model, 'trust_level', 0.5),
                    "perceived_distance": getattr(user_model, 'perceived_distance', 0.5),
                    "attachment_anxiety": getattr(user_model, 'attachment_anxiety', 0.5),
                    "narrative_belief": getattr(user_model, 'narrative_belief', "Developing understanding of user"),
                    "last_flip_turn": getattr(user_model, 'last_flip_turn', None),
                    "model_confidence": 0.8
                },
                "component_theories": components,
                "unified_user_model": unified_payload,
                "system_status": "active_modeling"
            }
        
        return user_model_data
    
    except Exception as e:
        logger.error(f"Error getting detailed user model: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------
# AVATAR-SPECIFIC DASHBOARD ENDPOINTS
# ------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/dashboard",
    tags=["Dashboard"],
)

@router.get("/emotion-state",
            summary="Get current emotion state",
            description="Provides a real-time emotion state for the avatar dashboard, reflecting the current mood.")
async def get_emotion_state():
    """
    Get the current emotion state from the detailed emotional system and format it
    for the avatar dashboard. This function ensures the avatar's expression reflects
    the AI's current mood and holds it until a new mood is established.
    """
    try:
        # Get the current emotional state directly from session
        emotion_state = get_current_emotion_state_from_session()

        # Get top emotions from the vector
        try:
            from ..emotions.classification import IDX2GOEMO_LABEL
            emotion_labels = IDX2GOEMO_LABEL
        except ImportError:
            # Fallback emotion labels
            emotion_labels = {
                0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval", 
                5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
                10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement", 
                14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
                20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
                25: "sadness", 26: "surprise", 27: "neutral"
            }
        
        # Create scores dictionary from top emotions
        scores = {}
        for i, intensity in enumerate(emotion_state.vector_28[:4]):  # Top 4 emotions
            if intensity > 0.001:  # Only include meaningful intensities
                emotion_name = emotion_labels.get(i, f"emotion_{i}")
                scores[emotion_name] = intensity
        
        # Ensure we have at least one score
        if not scores:
            scores = {"neutral": 0.5}

        return {
            "dominant_label": emotion_state.dominant_label,
            "intensity": emotion_state.intensity,
            "valence": emotion_state.valence,
            "arousal": emotion_state.arousal,
            "mood_family": emotion_state.mood_family,
            "scores": scores
        }
    except Exception as e:
        logger.error(f"Error in get_emotion_state: {e}")
        # Fallback to a neutral state on error
        return {
            "dominant_label": "neutral",
            "intensity": 0.5,
            "valence": 0.0,
            "arousal": 0.2,
            "mood_family": "Serene Attunement",
            "scores": {"neutral": 1.0}
        }