import logging
import asyncio
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from ..config import (
    embedder, classifier, chroma_db, neo4j_conn, 
    ACTIVE_SESSION_ID, CONVERSATION_SESSIONS, 
    estimate_token_count, get_system_health,
    THINKING_LAYER_ENABLED, THINKING_MAX_TIME, THINKING_DEPTH_THRESHOLD, THINKING_DEBUG_LOGGING
)
from ..memory import (
    get_memory_stats
)
from ..models import ConversationSession, ChatRequest
from .chat_endpoints import chat
from ..adaptive_language import get_mood_state
from ..config import meta_architecture_analyzer
from ..conversations.session_manager import set_active_session as set_active_session_logic

router = APIRouter(
    prefix="/v1/avatar",
    tags=["Avatar"],
)

logger = logging.getLogger(__name__)

@router.get("/sessions", summary="Get all conversation sessions")
async def get_sessions():
    sessions = [session.dict() for session in CONVERSATION_SESSIONS.values()]
    return {"sessions": sessions, "active_session_id": ACTIVE_SESSION_ID}

@router.post("/sessions/new", summary="Create a new conversation session")
async def create_session():
    global ACTIVE_SESSION_ID
    session_id = str(uuid.uuid4())
    new_session = ConversationSession(
        session_id=session_id,
        title="New Conversation",
        created_at=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    CONVERSATION_SESSIONS[session_id] = new_session
    ACTIVE_SESSION_ID = session_id
    return new_session.dict()

@router.delete("/sessions/{session_id}", summary="Delete a conversation session")
async def delete_session(session_id: str):
    global ACTIVE_SESSION_ID
    if session_id in CONVERSATION_SESSIONS:
        del CONVERSATION_SESSIONS[session_id]
        if ACTIVE_SESSION_ID == session_id:
            if CONVERSATION_SESSIONS:
                ACTIVE_SESSION_ID = list(CONVERSATION_SESSIONS.keys())[0]
            else:
                ACTIVE_SESSION_ID = None
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.post("/sessions/{session_id}/set_active", summary="Set the active conversation session")
async def set_active_session(session_id: str):
    success = await set_active_session_logic(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "active_session_id": session_id}

@router.post("/sessions/{session_id}/chat", summary="Send a chat message to a session")
async def avatar_chat(session_id: str, req: ChatRequest, bg: BackgroundTasks):
    return await chat(req, bg)

@router.get("/sessions/{session_id}", summary="Get a specific conversation session")
async def get_session(session_id: str):
    if session_id in CONVERSATION_SESSIONS:
        return CONVERSATION_SESSIONS[session_id].dict()
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/mood/current", summary="Get current mood")
async def get_current_mood():
    return get_mood_state()

@router.get("/consciousness/current", summary="Get current consciousness state")
async def get_consciousness_state():
    if meta_architecture_analyzer:
        return await meta_architecture_analyzer.get_consciousness_analysis()
    else:
        return {"error": "Meta architecture analyzer not available"}