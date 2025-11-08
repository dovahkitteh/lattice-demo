import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ..config import CONVERSATION_SESSIONS
from ..models import ConversationAnalysis, ConversationMessage, ConversationSession
from ..emotions import get_top_emotions, get_emotion_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONVERSATION ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------

async def analyze_conversation_session(session_id: str) -> Optional[ConversationAnalysis]:
    """Perform comprehensive analysis of a conversation session"""
    if session_id not in CONVERSATION_SESSIONS:
        logger.error(f"Session {session_id} not found for analysis")
        return None
    
    session = CONVERSATION_SESSIONS[session_id]
    
    try:
        # Analyze user behavioral patterns
        user_patterns = analyze_user_behavior(session)
        
        # Analyze AI performance
        ai_performance = analyze_ai_performance(session)
        
        # Create conversation summary
        conversation_summary = create_detailed_summary(session)
        
        # Extract key insights
        insights = extract_key_insights(session)
        
        # Generate improvement suggestions
        improvements = generate_improvement_suggestions(session)
        
        # Analyze emotional journey
        emotional_journey = analyze_emotional_journey(session)
        
        # Calculate training value
        training_value = calculate_training_value(session)
        
        # Get memory nodes created (placeholder - would need to track this)
        memory_nodes = []  # This would be populated by tracking memory creation
        
        analysis = ConversationAnalysis(
            session_id=session_id,
            user_behavioral_patterns=user_patterns,
            ai_performance_metrics=ai_performance,
            conversation_summary=conversation_summary,
            key_insights=insights,
            improvement_suggestions=improvements,
            emotional_journey=emotional_journey,
            training_value_score=training_value,
            memory_nodes_created=memory_nodes
        )
        
        # Mark session as analyzed
        session.analysis_complete = True
        session.last_updated = datetime.now(timezone.utc)
        
        logger.info(f"✅ Completed analysis for session {session_id[:8]}")
        return analysis
    
    except Exception as e:
        logger.error(f"❌ Error analyzing session {session_id[:8]}: {e}")
        return None

def analyze_user_behavior(session: ConversationSession) -> Dict[str, Any]:
    """Analyze user behavioral patterns in the conversation"""
    user_messages = [msg for msg in session.messages if msg.role == "user"]
    
    if not user_messages:
        return {"error": "No user messages to analyze"}
    
    # Message length analysis
    message_lengths = [len(msg.content) for msg in user_messages]
    avg_length = sum(message_lengths) / len(message_lengths)
    
    # Timing analysis
    if len(user_messages) > 1:
        response_times = []
        for i in range(1, len(user_messages)):
            time_diff = user_messages[i].timestamp - user_messages[i-1].timestamp
            response_times.append(time_diff.total_seconds())
        
        avg_response_time = sum(response_times) / len(response_times)
    else:
        avg_response_time = 0
    
    # Emotional patterns
    emotional_patterns = []
    for msg in user_messages:
        if msg.user_affect:
            emotions = get_top_emotions(msg.user_affect)
            emotional_patterns.extend(emotions)
    
    # Count emotional frequencies
    emotion_counts = {}
    for emotion in emotional_patterns:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Interaction patterns
    question_count = sum(1 for msg in user_messages if '?' in msg.content)
    exclamation_count = sum(1 for msg in user_messages if '!' in msg.content)
    
    return {
        "message_count": len(user_messages),
        "average_message_length": avg_length,
        "message_length_range": [min(message_lengths), max(message_lengths)],
        "average_response_time_seconds": avg_response_time,
        "dominant_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3],
        "question_frequency": question_count / len(user_messages),
        "exclamation_frequency": exclamation_count / len(user_messages),
        "engagement_level": calculate_engagement_level(user_messages),
        "conversation_style": determine_conversation_style(user_messages)
    }

def analyze_ai_performance(session: ConversationSession) -> Dict[str, Any]:
    """Analyze AI performance metrics"""
    ai_messages = [msg for msg in session.messages if msg.role == "assistant"]
    
    if not ai_messages:
        return {"error": "No AI messages to analyze"}
    
    # Response quality metrics
    response_lengths = [len(msg.content) for msg in ai_messages]
    avg_response_length = sum(response_lengths) / len(response_lengths)
    
    # Emotional responsiveness
    emotional_responsiveness = 0
    if len(ai_messages) > 0:
        emotional_responses = sum(1 for msg in ai_messages if msg.self_affect and sum(abs(x) for x in msg.self_affect) > 0.5)
        emotional_responsiveness = emotional_responses / len(ai_messages)
    
    # Consistency metrics
    consistency_score = calculate_response_consistency(ai_messages)
    
    # Helpfulness indicators
    helpfulness_score = calculate_helpfulness_score(ai_messages)
    
    return {
        "response_count": len(ai_messages),
        "average_response_length": avg_response_length,
        "response_length_consistency": calculate_length_consistency(response_lengths),
        "emotional_responsiveness": emotional_responsiveness,
        "consistency_score": consistency_score,
        "helpfulness_score": helpfulness_score,
        "response_quality_rating": calculate_overall_quality(ai_messages)
    }

def create_detailed_summary(session: ConversationSession) -> str:
    """Create a detailed summary of the conversation"""
    if not session.messages:
        return "Empty conversation"
    
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"Conversation: {session.title}")
    summary_parts.append(f"Duration: {session.last_updated - session.created_at}")
    summary_parts.append(f"Messages: {len(session.messages)} ({session.total_tokens} tokens)")
    
    # Participants
    user_msg_count = sum(1 for msg in session.messages if msg.role == "user")
    ai_msg_count = sum(1 for msg in session.messages if msg.role == "assistant")
    summary_parts.append(f"User messages: {user_msg_count}, AI messages: {ai_msg_count}")
    
    # Key topics (extracted from message content)
    key_topics = extract_key_topics(session.messages)
    if key_topics:
        summary_parts.append(f"Key topics: {', '.join(key_topics[:5])}")
    
    # Emotional arc
    emotional_arc = describe_emotional_arc(session.messages)
    if emotional_arc:
        summary_parts.append(f"Emotional arc: {emotional_arc}")
    
    return "\n".join(summary_parts)

def extract_key_insights(session: ConversationSession) -> List[str]:
    """Extract key insights from the conversation"""
    insights = []
    
    # Analyze conversation depth
    if len(session.messages) > 10:
        insights.append("Extended conversation showing sustained engagement")
    
    # Analyze emotional patterns
    user_emotions = []
    ai_emotions = []
    
    for msg in session.messages:
        if msg.role == "user" and msg.user_affect:
            user_emotions.extend(get_top_emotions(msg.user_affect))
        elif msg.role == "assistant" and msg.self_affect:
            ai_emotions.extend(get_top_emotions(msg.self_affect))
    
    if user_emotions:
        dominant_user_emotion = max(set(user_emotions), key=user_emotions.count)
        insights.append(f"User showed predominantly {dominant_user_emotion} throughout conversation")
    
    if ai_emotions:
        dominant_ai_emotion = max(set(ai_emotions), key=ai_emotions.count)
        insights.append(f"AI demonstrated {dominant_ai_emotion} responses")
    
    # Analyze conversation quality
    avg_msg_length = sum(len(msg.content) for msg in session.messages) / len(session.messages)
    if avg_msg_length > 200:
        insights.append("High-quality conversation with detailed exchanges")
    
    return insights

def generate_improvement_suggestions(session: ConversationSession) -> List[str]:
    """Generate suggestions for improvement"""
    suggestions = []
    
    # Analyze response times and suggest improvements
    user_messages = [msg for msg in session.messages if msg.role == "user"]
    ai_messages = [msg for msg in session.messages if msg.role == "assistant"]
    
    if len(ai_messages) > 0:
        avg_ai_length = sum(len(msg.content) for msg in ai_messages) / len(ai_messages)
        if avg_ai_length < 50:
            suggestions.append("Consider providing more detailed responses")
        elif avg_ai_length > 500:
            suggestions.append("Consider more concise responses for better engagement")
    
    # Emotional responsiveness
    emotional_ai_responses = sum(1 for msg in ai_messages if msg.self_affect and sum(abs(x) for x in msg.self_affect) > 0.5)
    if emotional_ai_responses < len(ai_messages) * 0.3:
        suggestions.append("Increase emotional responsiveness and empathy")
    
    # Conversation flow
    if len(session.messages) < 4:
        suggestions.append("Encourage longer conversations for better understanding")
    
    return suggestions

def analyze_emotional_journey(session: ConversationSession) -> Dict[str, Any]:
    """Analyze the emotional journey throughout the conversation"""
    emotional_timeline = []
    
    for msg in session.messages:
        if msg.user_affect or msg.self_affect:
            entry = {
                "timestamp": msg.timestamp.isoformat(),
                "role": msg.role,
                "message_excerpt": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            }
            
            if msg.user_affect:
                entry["user_emotions"] = get_top_emotions(msg.user_affect)
                entry["user_intensity"] = sum(abs(x) for x in msg.user_affect)
            
            if msg.self_affect:
                entry["ai_emotions"] = get_top_emotions(msg.self_affect)
                entry["ai_intensity"] = sum(abs(x) for x in msg.self_affect)
            
            emotional_timeline.append(entry)
    
    # Calculate emotional trajectory
    if len(emotional_timeline) > 1:
        start_intensity = emotional_timeline[0].get("user_intensity", 0)
        end_intensity = emotional_timeline[-1].get("user_intensity", 0)
        
        if end_intensity > start_intensity * 1.2:
            trajectory = "intensifying"
        elif end_intensity < start_intensity * 0.8:
            trajectory = "diminishing"
        else:
            trajectory = "stable"
    else:
        trajectory = "insufficient_data"
    
    return {
        "emotional_timeline": emotional_timeline,
        "trajectory": trajectory,
        "emotional_moments": len(emotional_timeline),
        "peak_intensity": max((entry.get("user_intensity", 0) for entry in emotional_timeline), default=0)
    }

def calculate_training_value(session: ConversationSession) -> float:
    """Calculate the training value of the conversation"""
    score = 0.0
    
    # Length bonus
    if len(session.messages) > 4:
        score += 0.2
    if len(session.messages) > 10:
        score += 0.2
    
    # Emotional richness
    emotional_messages = sum(1 for msg in session.messages if 
                           (msg.user_affect and sum(abs(x) for x in msg.user_affect) > 0.5) or
                           (msg.self_affect and sum(abs(x) for x in msg.self_affect) > 0.5))
    
    if emotional_messages > 0:
        emotional_ratio = emotional_messages / len(session.messages)
        score += emotional_ratio * 0.3
    
    # Diversity of content
    unique_words = set()
    for msg in session.messages:
        unique_words.update(msg.content.lower().split())
    
    if len(unique_words) > 50:
        score += 0.2
    
    # Conversation quality
    avg_length = sum(len(msg.content) for msg in session.messages) / len(session.messages)
    if avg_length > 100:
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

# Helper functions
def calculate_engagement_level(messages: List[ConversationMessage]) -> str:
    """Calculate user engagement level"""
    if not messages:
        return "unknown"
    
    avg_length = sum(len(msg.content) for msg in messages) / len(messages)
    question_ratio = sum(1 for msg in messages if '?' in msg.content) / len(messages)
    
    if avg_length > 100 and question_ratio > 0.3:
        return "high"
    elif avg_length > 50 or question_ratio > 0.2:
        return "medium"
    else:
        return "low"

def determine_conversation_style(messages: List[ConversationMessage]) -> str:
    """Determine the conversation style"""
    if not messages:
        return "unknown"
    
    total_questions = sum(1 for msg in messages if '?' in msg.content)
    total_exclamations = sum(1 for msg in messages if '!' in msg.content)
    avg_length = sum(len(msg.content) for msg in messages) / len(messages)
    
    if total_questions > len(messages) * 0.4:
        return "inquisitive"
    elif total_exclamations > len(messages) * 0.3:
        return "expressive"
    elif avg_length > 200:
        return "detailed"
    else:
        return "casual"

def calculate_response_consistency(messages: List[ConversationMessage]) -> float:
    """Calculate consistency of AI responses"""
    if len(messages) < 2:
        return 1.0
    
    lengths = [len(msg.content) for msg in messages]
    avg_length = sum(lengths) / len(lengths)
    
    # Calculate coefficient of variation
    variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
    std_dev = variance ** 0.5
    
    if avg_length == 0:
        return 1.0
    
    cv = std_dev / avg_length
    consistency = max(0, 1 - cv)  # Higher consistency = lower coefficient of variation
    
    return consistency

def calculate_helpfulness_score(messages: List[ConversationMessage]) -> float:
    """Calculate helpfulness score based on response characteristics"""
    if not messages:
        return 0.0
    
    score = 0.0
    
    # Check for helpful patterns
    for msg in messages:
        content = msg.content.lower()
        
        # Positive indicators
        if any(phrase in content for phrase in ['here are', 'i can help', 'let me', 'sure', 'of course']):
            score += 0.1
        
        # Detailed responses
        if len(msg.content) > 150:
            score += 0.1
        
        # Questions to clarify
        if '?' in msg.content:
            score += 0.05
    
    return min(score, 1.0)

def calculate_overall_quality(messages: List[ConversationMessage]) -> float:
    """Calculate overall response quality"""
    if not messages:
        return 0.0
    
    # Combine multiple factors
    length_score = min(sum(len(msg.content) for msg in messages) / (len(messages) * 100), 1.0)
    consistency_score = calculate_response_consistency(messages)
    helpfulness_score = calculate_helpfulness_score(messages)
    
    # Weighted average
    quality = (length_score * 0.4 + consistency_score * 0.3 + helpfulness_score * 0.3)
    
    return quality

def calculate_length_consistency(lengths: List[int]) -> float:
    """Calculate consistency of response lengths"""
    if len(lengths) < 2:
        return 1.0
    
    avg_length = sum(lengths) / len(lengths)
    variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
    
    # Normalize by average length
    normalized_variance = variance / (avg_length ** 2) if avg_length > 0 else 0
    
    # Convert to consistency score (0-1, higher is more consistent)
    consistency = max(0, 1 - normalized_variance)
    
    return consistency

def extract_key_topics(messages: List[ConversationMessage]) -> List[str]:
    """Extract key topics from conversation messages"""
    # Simple keyword extraction (could be enhanced with NLP)
    word_freq = {}
    
    for msg in messages:
        words = msg.content.lower().split()
        for word in words:
            if len(word) > 4 and word.isalpha():  # Filter meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top topics
    sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, count in sorted_topics[:10] if count > 1]

def describe_emotional_arc(messages: List[ConversationMessage]) -> str:
    """Describe the emotional arc of the conversation"""
    emotional_points = []
    
    for msg in messages:
        if msg.user_affect:
            intensity = sum(abs(x) for x in msg.user_affect)
            emotional_points.append(intensity)
    
    if len(emotional_points) < 2:
        return "Insufficient emotional data"
    
    # Analyze trend
    start_avg = sum(emotional_points[:2]) / 2
    end_avg = sum(emotional_points[-2:]) / 2
    
    if end_avg > start_avg * 1.3:
        return "Emotional intensity increased"
    elif end_avg < start_avg * 0.7:
        return "Emotional intensity decreased"
    else:
        return "Emotional intensity remained stable"

async def generate_training_data_for_session(session_id: str) -> dict:
    """Generate training data from a conversation session"""
    if session_id not in CONVERSATION_SESSIONS:
        return {"error": "Session not found"}
    
    session = CONVERSATION_SESSIONS[session_id]
    
    # Create training data directory
    training_dir = "./data/training_data"
    os.makedirs(training_dir, exist_ok=True)
    
    # Generate training file
    training_file = f"{training_dir}/conversation_{session_id[:8]}.jsonl"
    
    training_examples = []
    
    # Convert conversation to training format
    for i in range(len(session.messages) - 1):
        if session.messages[i].role == "user" and session.messages[i+1].role == "assistant":
            user_msg = session.messages[i].content
            ai_msg = session.messages[i+1].content
            
            # Create training example
            training_example = {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": ai_msg}
                ],
                "metadata": {
                    "session_id": session_id,
                    "timestamp": session.messages[i].timestamp.isoformat(),
                    "user_emotions": get_top_emotions(session.messages[i].user_affect) if session.messages[i].user_affect else [],
                    "ai_emotions": get_top_emotions(session.messages[i+1].self_affect) if session.messages[i+1].self_affect else []
                }
            }
            training_examples.append(training_example)
    
    # Write training data file
    with open(training_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    # Update session flag
    session.training_data_generated = True
    
    logger.info(f"Generated {len(training_examples)} training examples for session {session_id[:8]} → {training_file}")
    
    return {
        "session_id": session_id,
        "training_file": training_file,
        "examples_generated": len(training_examples),
        "file_size_kb": os.path.getsize(training_file) / 1024 if os.path.exists(training_file) else 0
    } 