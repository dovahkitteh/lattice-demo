import logging
from typing import List
from datetime import datetime, timezone

from ..config import CONVERSATION_SESSIONS, MAX_CONVERSATION_TOKENS, COMPRESSION_THRESHOLD
from ..models import ConversationMessage, ConversationSession
from ..memory import store_smg_node_smart
from ..emotions import classify_affect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONVERSATION COMPRESSION FUNCTIONS
# ---------------------------------------------------------------------------

async def compress_conversation_if_needed(session_id: str) -> bool:
    """Compress conversation if it exceeds token limits"""
    if session_id not in CONVERSATION_SESSIONS:
        return False
    
    session = CONVERSATION_SESSIONS[session_id]
    
    # Check if compression is needed
    if session.total_tokens < MAX_CONVERSATION_TOKENS * COMPRESSION_THRESHOLD:
        return False
    
    logger.info(f"🗜️ Starting compression for session {session_id[:8]} ({session.total_tokens} tokens)")
    
    # Get messages to compress (keep last 5 messages, compress older ones)
    messages_to_keep = session.messages[-5:]  # Keep recent messages
    messages_to_compress = session.messages[:-5]  # Compress older messages
    
    if not messages_to_compress:
        logger.info("📝 No messages to compress")
        return False
    
    # Create compression summary
    compression_summary = await create_conversation_summary(messages_to_compress)
    
    # Store compression summary as a memory node
    affect_vec = await classify_affect(compression_summary)
    # Use smart storage that switches between unified and legacy based on feature flags
    await store_smg_node_smart(
        msg=compression_summary,
        affect_vec=affect_vec,
        synopsis=f"Compressed conversation summary for session {session_id[:8]}",
        reflection=f"Conversation compressed from {len(messages_to_compress)} messages to summary",
        origin="conversation_compression"
    )
    
    # Update session with compressed messages
    session.messages = messages_to_keep
    session.total_tokens = sum(msg.token_count for msg in messages_to_keep)
    session.last_updated = datetime.now(timezone.utc)
    
    logger.info(f"✅ Compressed {len(messages_to_compress)} messages, kept {len(messages_to_keep)} recent messages")
    return True

async def create_conversation_summary(messages: List[ConversationMessage]) -> str:
    """Create a summary of conversation messages"""
    if not messages:
        return "Empty conversation"
    
    # Group messages by user/assistant pairs
    conversation_pairs = []
    current_user_msg = None
    
    for msg in messages:
        if msg.role == "user":
            current_user_msg = msg.content
        elif msg.role == "assistant" and current_user_msg:
            conversation_pairs.append({
                "user": current_user_msg,
                "assistant": msg.content,
                "timestamp": msg.timestamp
            })
            current_user_msg = None
    
    # Create summary
    if not conversation_pairs:
        return "Conversation without complete exchanges"
    
    summary_parts = []
    summary_parts.append(f"Conversation Summary ({len(conversation_pairs)} exchanges):")
    
    # Add temporal context
    start_time = messages[0].timestamp
    end_time = messages[-1].timestamp
    duration = end_time - start_time
    summary_parts.append(f"Duration: {duration.total_seconds():.0f} seconds")
    
    # Add key exchanges (sample from beginning, middle, end)
    key_exchanges = []
    if len(conversation_pairs) > 0:
        key_exchanges.append(("Beginning", conversation_pairs[0]))
    if len(conversation_pairs) > 2:
        mid_idx = len(conversation_pairs) // 2
        key_exchanges.append(("Middle", conversation_pairs[mid_idx]))
    if len(conversation_pairs) > 1:
        key_exchanges.append(("End", conversation_pairs[-1]))
    
    for period, exchange in key_exchanges:
        summary_parts.append(f"\n{period} Exchange:")
        summary_parts.append(f"User: {exchange['user'][:100]}...")
        summary_parts.append(f"Assistant: {exchange['assistant'][:100]}...")
    
    # Add emotional context if available
    emotional_moments = []
    for msg in messages:
        if msg.user_affect:
            user_affect_magnitude = sum(abs(x) for x in msg.user_affect)
            if user_affect_magnitude > 1.5:  # Significant emotional moment
                emotional_moments.append(f"High user emotion: {msg.content[:50]}...")
        
        if msg.self_affect:
            self_affect_magnitude = sum(abs(x) for x in msg.self_affect)
            if self_affect_magnitude > 1.5:  # Significant emotional moment
                emotional_moments.append(f"High AI emotion: {msg.content[:50]}...")
    
    if emotional_moments:
        summary_parts.append(f"\nEmotional Highlights ({len(emotional_moments)}):")
        for moment in emotional_moments[:3]:  # Top 3 emotional moments
            summary_parts.append(f"- {moment}")
    
    return "\n".join(summary_parts)

async def get_conversation_compression_stats(session_id: str) -> dict:
    """Get compression statistics for a conversation"""
    if session_id not in CONVERSATION_SESSIONS:
        return {"error": "Session not found"}
    
    session = CONVERSATION_SESSIONS[session_id]
    
    # Calculate compression metrics
    current_tokens = session.total_tokens
    max_tokens = MAX_CONVERSATION_TOKENS
    compression_threshold_tokens = int(max_tokens * COMPRESSION_THRESHOLD)
    
    compression_needed = current_tokens >= compression_threshold_tokens
    tokens_until_compression = max(0, compression_threshold_tokens - current_tokens)
    
    return {
        "session_id": session_id,
        "current_tokens": current_tokens,
        "max_tokens": max_tokens,
        "compression_threshold": compression_threshold_tokens,
        "compression_needed": compression_needed,
        "tokens_until_compression": tokens_until_compression,
        "utilization_percentage": (current_tokens / max_tokens) * 100,
        "messages_count": len(session.messages),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def estimate_compression_benefit(messages: List[ConversationMessage]) -> dict:
    """Estimate the benefit of compressing messages"""
    if not messages:
        return {"error": "No messages to analyze"}
    
    # Calculate current token usage
    current_tokens = sum(msg.token_count for msg in messages)
    
    # Estimate compression savings (typical 70-80% reduction)
    estimated_summary_tokens = max(100, int(current_tokens * 0.1))  # 10% of original
    estimated_savings = current_tokens - estimated_summary_tokens
    
    # Calculate time span
    time_span = messages[-1].timestamp - messages[0].timestamp
    
    return {
        "original_tokens": current_tokens,
        "estimated_summary_tokens": estimated_summary_tokens,
        "estimated_savings": estimated_savings,
        "compression_ratio": estimated_savings / current_tokens if current_tokens > 0 else 0,
        "messages_count": len(messages),
        "time_span_seconds": time_span.total_seconds(),
        "compression_worthwhile": estimated_savings > 500  # Threshold for worthwhile compression
    } 