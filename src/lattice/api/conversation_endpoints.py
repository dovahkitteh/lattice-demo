import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

# ACTIVE_SESSION_ID imported from session_manager to ensure sync
from ..conversations import (
    get_all_sessions, get_session_details, analyze_conversation_session,
    delete_session, set_active_session
)
from ..conversations.session_manager import create_new_session, CONVERSATION_SESSIONS
from ..models import ConversationSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONVERSATION MANAGEMENT ENDPOINTS
# ---------------------------------------------------------------------------

async def get_conversation_sessions():
    """Get all available conversation sessions"""
    try:
        sessions_data = get_all_sessions()
        # Return the sessions array directly, not wrapped in another "sessions" key
        return sessions_data
    except Exception as e:
        logger.error(f"Error getting conversation sessions: {e}")
        return {"error": str(e)}

async def get_conversation_session(session_id: str):
    """Get details of a specific conversation session"""
    details = get_session_details(session_id)
    if "error" in details:
        raise HTTPException(status_code=404, detail=details["error"])
    return details

async def delete_conversation_session(session_id: str):
    """Delete a conversation session"""
    success = await delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}

async def set_active_conversation_session(session_id: str):
    """Set the active conversation session"""
    success = await set_active_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    # If the session has a persisted emotional state, update dashboard cache
    try:
        from ..conversations.session_manager import CONVERSATION_SESSIONS
        from ..config import DASHBOARD_STATE_CACHE
        session = CONVERSATION_SESSIONS.get(session_id)
        if session and getattr(session, "emotion_state", None):
            # Also propagate user_model if present
            DASHBOARD_STATE_CACHE.update({
                "current_emotion_state": session.emotion_state,
                "current_user_model": getattr(session, "user_model", None),
                "session_id": session_id,
            })
            logger.info("🎭 Restored dashboard emotion state from session switch")
    except Exception as e:
        logger.warning(f"Failed to update dashboard cache on session switch: {e}")

    return {"status": "active", "session_id": session_id}

async def create_conversation_session(request: dict):
    """Create a new conversation session"""
    try:
        first_message = request.get("first_message", "")
        set_as_active = request.get("set_as_active", True)
        
        # Use the proper session creation function
        session_id = await create_new_session(first_message, set_as_active)
        
        return {"status": "created", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating conversation session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def end_conversation_session_endpoint(session_id: str):
    """End a conversation session"""
    try:
        from ..conversations import end_conversation_session
        success = await end_conversation_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ended", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error ending conversation session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_conversation_analysis(session_id: str):
    """Get analysis of a conversation session"""
    try:
        analysis = await analyze_conversation_session(session_id)
        return {
            "session_id": session_id,
            "analysis": analysis,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting conversation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_active_session():
    """Get the active conversation session"""
    try:
        from .. import config
        return {"active_session": config.ACTIVE_SESSION_ID}
    except Exception as e:
        logger.error(f"Error getting active session: {e}")
        return {"error": str(e)}

async def get_live_recursive_analysis(session_id: str):
    """Get live recursive analysis for a session"""
    try:
        from ..conversations.turn_analyzer import turn_analyzer
        analysis = turn_analyzer.get_session_analysis(session_id)
        return {"session_id": session_id, "analysis": analysis}
    except Exception as e:
        logger.error(f"Error getting live recursive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_training_data_endpoint(session_id: str):
    """Generate training data from a conversation session"""
    try:
        from ..conversations import generate_training_data_for_session
        training_data = await generate_training_data_for_session(session_id)
        return {
            "session_id": session_id,
            "training_data": training_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))