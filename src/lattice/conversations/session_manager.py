import uuid
import logging
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict

from ..config import (
    CONVERSATION_SESSIONS, estimate_token_count, 
    generate_session_title, LATTICE_DB_PATH
)
from .. import config
from ..models import ConversationSession, ConversationMessage, Message

logger = logging.getLogger(__name__)

SESSIONS_DIR = os.path.join(os.path.dirname(LATTICE_DB_PATH), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SESSION PERSISTENCE
# ---------------------------------------------------------------------------

def save_session(session: ConversationSession):
    """Save a conversation session to a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{session.session_id}.json")
    try:
        with open(session_path, 'w', encoding='utf-8') as f:
            # Use model_dump instead of dict() for better Pydantic v2 compatibility
            session_data = session.model_dump(mode='json')
            json.dump(session_data, f, indent=4, default=str)
        logger.debug(f"💾 Saved session {session.session_id[:8]} to disk.")
    except Exception as e:
        logger.error(f"Error saving session {session.session_id}: {e}")

def load_session(session_id: str) -> Optional[ConversationSession]:
    """Load a single session from a JSON file."""
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        return None
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Fix legacy data format issues
            if 'emotion_state' in data and data['emotion_state']:
                emotion_state = data['emotion_state']
                
                # Fix flags field: convert string representation of set to list
                if 'flags' in emotion_state:
                    flags_value = emotion_state['flags']
                    if isinstance(flags_value, str) and flags_value == "set()":
                        emotion_state['flags'] = []
                    elif isinstance(flags_value, str):
                        # Try to handle other string representations
                        emotion_state['flags'] = []
                
                # Ensure all required fields exist with defaults
                emotion_state.setdefault('valence', 0.0)
                emotion_state.setdefault('arousal', 0.0)
                emotion_state.setdefault('attachment_security', 0.5)
                emotion_state.setdefault('intensity', 0.0)
                emotion_state.setdefault('mood_family', 'Serene Attunement')
            
            return ConversationSession(**data)
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None

def load_all_sessions():
    """Load all sessions from the sessions directory into memory."""
    global CONVERSATION_SESSIONS
    if CONVERSATION_SESSIONS: # Avoid reloading if already populated
        logger.info("Sessions already loaded.")
        return

    logger.info(f"💾 Loading conversation sessions from {SESSIONS_DIR}...")
    loaded_count = 0
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            session = load_session(session_id)
            if session:
                CONVERSATION_SESSIONS[session_id] = session
                loaded_count += 1
    logger.info(f"✅ Loaded {loaded_count} sessions into memory.")

# ---------------------------------------------------------------------------
# CONVERSATION SESSION MANAGEMENT
# ---------------------------------------------------------------------------

async def create_new_session(first_message: str = "", set_as_active: bool = False) -> str:
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    title = generate_session_title(first_message) if first_message else "New Conversation"
    
    session = ConversationSession(
        session_id=session_id,
        title=title,
        created_at=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc)
    )
    
    CONVERSATION_SESSIONS[session_id] = session
    
    # Only set as active if explicitly requested
    if set_as_active:
        config.ACTIVE_SESSION_ID = session_id
    
    save_session(session)
    
    logger.info(f"🆕 Created new conversation session: {session_id[:8]} - '{title}' (active: {set_as_active})")
    return session_id

async def add_message_to_session(session_id: str, role: str, content: str, 
                                user_affect: Optional[List[float]] = None,
                                self_affect: Optional[List[float]] = None) -> str:
    """Add a message to a conversation session"""
    if session_id not in CONVERSATION_SESSIONS:
        logger.error(f"Session {session_id} not found")
        return ""
    
    session = CONVERSATION_SESSIONS[session_id]
    message_id = str(uuid.uuid4())
    token_count = estimate_token_count(content)
    
    message = ConversationMessage(
        id=message_id,
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
        user_affect=user_affect,
        self_affect=self_affect,
        token_count=token_count
    )
    
    session.messages.append(message)
    session.total_tokens += token_count
    session.last_updated = datetime.now(timezone.utc)
    save_session(session)
    
    logger.info(f"💬 Added {role} message to session {session_id[:8]} (tokens: {token_count})")
    return message_id

async def get_or_create_active_session() -> str:
    """Get the active session or create a new one"""
    if config.ACTIVE_SESSION_ID and config.ACTIVE_SESSION_ID in CONVERSATION_SESSIONS:
        session = CONVERSATION_SESSIONS[config.ACTIVE_SESSION_ID]
        if session.is_active:
            return config.ACTIVE_SESSION_ID
    
    # Create new session if no active session exists, and set it as active
    return await create_new_session(set_as_active=True)

async def get_session_context_for_prompt(session_id: str) -> List[Message]:
    """Get session messages formatted for prompt building"""
    if session_id not in CONVERSATION_SESSIONS:
        return []
    
    session = CONVERSATION_SESSIONS[session_id]
    
    # Convert to Message format for prompt building
    messages = []
    for msg in session.messages:
        messages.append(Message(role=msg.role, content=msg.content))
    
    return messages

async def end_conversation_session(session_id: str, reason: str = "manual") -> bool:
    """End a conversation session"""
    if session_id not in CONVERSATION_SESSIONS:
        logger.error(f"Session {session_id} not found")
        return False
    
    session = CONVERSATION_SESSIONS[session_id]
    session.is_active = False
    session.ended_at = datetime.now(timezone.utc)
    session.end_reason = reason
    
    # Clear active session if this was the active one
    if config.ACTIVE_SESSION_ID == session_id:
        config.ACTIVE_SESSION_ID = None
    
    save_session(session)
    logger.info(f"🔚 Ended conversation session {session_id[:8]} (reason: {reason})")
    return True

async def set_active_session(session_id: str) -> bool:
    """Set a specific session as the active one."""
    if session_id not in CONVERSATION_SESSIONS:
        logger.error(f"Cannot set active session: Session {session_id} not found.")
        return False
    
    # Mark all other sessions as inactive
    for sid, session in CONVERSATION_SESSIONS.items():
        if sid != session_id and session.is_active:
            session.is_active = False
            save_session(session)
    
    # Mark the target session as active
    target_session = CONVERSATION_SESSIONS[session_id]
    target_session.is_active = True
    save_session(target_session)
    
    config.ACTIVE_SESSION_ID = session_id
    logger.info(f"🔄 Set active session to: {session_id[:8]}")
    return True

async def delete_session(session_id: str) -> bool:
    """Delete a conversation session permanently."""
    # Remove from memory
    if session_id in CONVERSATION_SESSIONS:
        del CONVERSATION_SESSIONS[session_id]
        logger.info(f"🗑️ Deleted session {session_id[:8]} from memory.")
    
    # Delete file from disk
    session_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(session_path):
        try:
            os.remove(session_path)
            logger.info(f"🗑️ Deleted session file: {session_path}")
        except Exception as e:
            logger.error(f"Error deleting session file {session_path}: {e}")
            return False # Indicate partial failure
            
    # If the deleted session was the active one, clear it
    if config.ACTIVE_SESSION_ID == session_id:
        config.ACTIVE_SESSION_ID = None
        logger.info("Cleared active session ID as it was deleted.")
        
    return True

def get_all_sessions() -> dict:
    """Get all conversation sessions"""
    sessions_data = []
    
    for session_id, session in CONVERSATION_SESSIONS.items():
        session_info = {
            "session_id": session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "message_count": len(session.messages),
            "total_tokens": session.total_tokens,
            "is_active": session.is_active,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "end_reason": session.end_reason,
            "analysis_complete": session.analysis_complete,
            "training_data_generated": session.training_data_generated
        }
        sessions_data.append(session_info)
    
    # Sort by last_updated (most recent first)
    sessions_data.sort(key=lambda x: x["last_updated"], reverse=True)
    
    return {
        "sessions": sessions_data,
        "total_sessions": len(sessions_data),
        "active_sessions": sum(1 for s in sessions_data if s["is_active"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def get_session_details(session_id: str) -> dict:
    """Get detailed information about a specific session"""
    if session_id not in CONVERSATION_SESSIONS:
        return {"error": "Session not found"}
    
    session = CONVERSATION_SESSIONS[session_id]
    
    # Get message details
    messages_data = []
    for msg in session.messages:
        msg_data = {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "token_count": msg.token_count
        }
        
        # Add affect information if available
        if msg.user_affect:
            msg_data["user_affect_magnitude"] = sum(abs(x) for x in msg.user_affect)
        if msg.self_affect:
            msg_data["self_affect_magnitude"] = sum(abs(x) for x in msg.self_affect)
        
        messages_data.append(msg_data)
    
    return {
        "session_id": session_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "last_updated": session.last_updated.isoformat(),
        "messages": messages_data,
        "message_count": len(messages_data),
        "total_tokens": session.total_tokens,
        "is_active": session.is_active,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "end_reason": session.end_reason,
        "analysis_complete": session.analysis_complete,
        "training_data_generated": session.training_data_generated
    } 