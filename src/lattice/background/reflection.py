import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from ..config import embedder, DEVICE
from ..emotions import classify_affect, classify_llm_affect, get_emotional_influence
from ..memory import update_node_with_self_affect_and_reflections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# REFLECTION GENERATION FUNCTIONS
# ---------------------------------------------------------------------------

async def generate_lightweight_reflection(original_message: str, channel: str = "general") -> str:
    """Generate a lightweight reflection for a message"""
    try:
        # Simple reflection patterns based on channel
        reflection_templates = {
            "general": [
                f"User expressed: {original_message[:50]}...",
                f"This interaction involved: {original_message[:30]}...",
                f"Key point raised: {original_message[:40]}..."
            ],
            "emotional": [
                f"Strong emotional expression detected: {original_message[:40]}...",
                f"User's emotional state influenced by: {original_message[:30]}...",
                f"Emotional resonance found in: {original_message[:35]}..."
            ],
            "technical": [
                f"Technical discussion about: {original_message[:40]}...",
                f"User sought information on: {original_message[:35]}...",
                f"Problem-solving session regarding: {original_message[:30]}..."
            ]
        }
        
        templates = reflection_templates.get(channel, reflection_templates["general"])
        
        # Use first template for simplicity
        reflection = templates[0]
        
        logger.debug(f"💭 Generated lightweight reflection for {channel} channel")
        return reflection
        
    except Exception as e:
        logger.error(f"❌ Error generating lightweight reflection: {e}")
        return f"Reflection on user message: {original_message[:50]}..."

async def generate_enhanced_self_affect(user_message: str, context: list[str]) -> list[float]:
    """Generate enhanced self-affect based on user message and context"""
    try:
        if not embedder:
            logger.warning("Embedder not available, generating basic self-affect")
            return [0.1] * 28  # Basic neutral affect
        
        # Create enhanced prompt for self-affect analysis
        context_summary = " ".join(context[:3])  # Use first 3 context items
        
        enhanced_prompt = f"""
        Context: {context_summary}
        User message: {user_message}
        
        My internal response to this interaction involves contemplation of the user's needs,
        consideration of helpful responses, and engagement with the topic at hand.
        """
        
        # Generate self-affect
        self_affect = await classify_llm_affect(enhanced_prompt)
        
        # Apply enhancement factors
        enhanced_affect = []
        for i, affect_value in enumerate(self_affect):
            # Enhance certain emotions based on context
            if i in [7, 13, 15, 20]:  # curiosity, excitement, gratitude, optimism
                enhanced_value = min(1.0, affect_value * 1.2)  # Boost positive engagement
            elif i in [2, 3, 10]:  # anger, annoyance, disapproval
                enhanced_value = affect_value * 0.8  # Reduce negative emotions
            else:
                enhanced_value = affect_value
                
            enhanced_affect.append(enhanced_value)
        
        logger.debug(f"🎭 Generated enhanced self-affect (magnitude: {sum(abs(x) for x in enhanced_affect):.3f})")
        return enhanced_affect
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced self-affect: {e}")
        return [0.1] * 28

async def generate_reflection(node_id: str, original_message: str, context: list[str], channel: str = "general") -> str:
    """Generate a comprehensive reflection for a conversation turn"""
    try:
        logger.debug(f"💭 Generating reflection for node {node_id[:8]}")
        
        # Analyze the message for key themes
        message_length = len(original_message)
        has_questions = '?' in original_message
        has_emotions = any(word in original_message.lower() for word in 
                          ['feel', 'think', 'believe', 'worry', 'hope', 'fear', 'love', 'hate'])
        
        # Build reflection based on analysis
        reflection_parts = []
        
        # Message analysis
        if message_length > 200:
            reflection_parts.append("User provided detailed input requiring thorough consideration")
        elif message_length < 50:
            reflection_parts.append("User sent concise message needing focused response")
        else:
            reflection_parts.append("User message was well-structured and clear")
        
        # Content analysis
        if has_questions:
            reflection_parts.append("inquiry-based interaction requiring informative response")
        
        if has_emotions:
            reflection_parts.append("emotional content detected, empathetic response appropriate")
        
        # Context integration
        if context:
            context_count = len(context)
            reflection_parts.append(f"drew from {context_count} contextual memories for response formation")
        else:
            reflection_parts.append("responded without significant prior context")
        
        # Channel-specific insights
        if channel == "emotional":
            reflection_parts.append("prioritized emotional intelligence and empathy")
        elif channel == "technical":
            reflection_parts.append("focused on accuracy and technical precision")
        elif channel == "creative":
            reflection_parts.append("engaged creative and imaginative response patterns")
        
        # Combine reflection parts
        reflection = "Reflection: " + "; ".join(reflection_parts) + "."
        
        logger.debug(f"💭 Generated comprehensive reflection for {node_id[:8]}")
        return reflection
        
    except Exception as e:
        logger.error(f"❌ Error generating reflection: {e}")
        return await generate_lightweight_reflection(original_message, channel)

async def reflect_on_turn(node_id: str, original_message: str, context: list[str]):
    """Generate and store reflection for a conversation turn"""
    try:
        # Generate user and self reflections
        user_reflection = await generate_user_reflection(original_message, context)
        self_reflection = await generate_reflection(node_id, original_message, context)
        
        # Generate enhanced self-affect
        self_affect = await generate_enhanced_self_affect(original_message, context)
        
        # Store all together
        await update_node_with_self_affect_and_reflections(
            node_id, self_affect, user_reflection, self_reflection
        )
        
        logger.info(f"💭 Completed reflection for turn {node_id[:8]}")
        
    except Exception as e:
        logger.error(f"❌ Error in turn reflection: {e}")

async def generate_user_reflection(original_message: str, context: list[str]) -> str:
    """Generate a reflection on the user's message and intent"""
    try:
        # Analyze user message characteristics
        message_analysis = analyze_user_message(original_message)
        
        # Build user reflection
        reflection_parts = []
        
        # Intent analysis
        if message_analysis["has_questions"]:
            reflection_parts.append("seeking information or clarification")
        
        if message_analysis["emotional_indicators"]:
            emotions = ", ".join(message_analysis["emotional_indicators"])
            reflection_parts.append(f"expressing {emotions}")
        
        if message_analysis["complexity_level"] == "high":
            reflection_parts.append("demonstrating sophisticated thinking")
        elif message_analysis["complexity_level"] == "low":
            reflection_parts.append("communicating with directness and clarity")
        
        # Context awareness
        if context:
            reflection_parts.append("building on previous conversation elements")
        else:
            reflection_parts.append("initiating new topic or direction")
        
        # Engagement level
        if message_analysis["engagement_level"] == "high":
            reflection_parts.append("showing strong engagement and interest")
        elif message_analysis["engagement_level"] == "low":
            reflection_parts.append("maintaining basic conversational participation")
        
        # Combine reflection
        if reflection_parts:
            user_reflection = "User appears to be " + ", ".join(reflection_parts) + "."
        else:
            user_reflection = "User's intent and state unclear from this message."
        
        logger.debug(f"👤 Generated user reflection: {user_reflection[:50]}...")
        return user_reflection
        
    except Exception as e:
        logger.error(f"❌ Error generating user reflection: {e}")
        return f"User expressed: {original_message[:100]}..."

def analyze_user_message(message: str) -> dict:
    """Analyze user message for various characteristics"""
    try:
        analysis = {
            "length": len(message),
            "word_count": len(message.split()),
            "has_questions": '?' in message,
            "has_exclamations": '!' in message,
            "emotional_indicators": [],
            "complexity_level": "medium",
            "engagement_level": "medium"
        }
        
        # Detect emotional indicators
        emotion_words = {
            "excitement": ["amazing", "awesome", "excited", "thrilled", "fantastic"],
            "concern": ["worried", "concerned", "anxious", "nervous", "unsure"],
            "satisfaction": ["great", "good", "pleased", "satisfied", "happy"],
            "frustration": ["frustrated", "annoyed", "stuck", "confused", "difficult"]
        }
        
        message_lower = message.lower()
        for emotion, words in emotion_words.items():
            if any(word in message_lower for word in words):
                analysis["emotional_indicators"].append(emotion)
        
        # Determine complexity level
        if analysis["word_count"] > 50 and any(indicator in message_lower for indicator in 
                                             ["because", "however", "therefore", "although", "moreover"]):
            analysis["complexity_level"] = "high"
        elif analysis["word_count"] < 10:
            analysis["complexity_level"] = "low"
        
        # Determine engagement level
        if (analysis["has_questions"] and analysis["word_count"] > 20) or analysis["emotional_indicators"]:
            analysis["engagement_level"] = "high"
        elif analysis["word_count"] < 5:
            analysis["engagement_level"] = "low"
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Error analyzing user message: {e}")
        return {"length": len(message), "word_count": len(message.split()), "has_questions": False}

async def generate_contextual_reflection(message: str, context: list[str], emotional_state: list[float]) -> str:
    """Generate reflection that considers context and emotional state"""
    try:
        # Analyze emotional influence
        emotional_influence = await get_emotional_influence(emotional_state)
        
        # Extract key context themes
        context_themes = extract_context_themes(context)
        
        # Build contextual reflection
        reflection_parts = [
            f"Message received in context of {len(context)} prior interactions"
        ]
        
        if context_themes:
            themes_str = ", ".join(context_themes[:3])  # Top 3 themes
            reflection_parts.append(f"with recurring themes: {themes_str}")
        
        # Add emotional context
        if emotional_state and sum(abs(x) for x in emotional_state) > 0.5:
            reflection_parts.append(f"emotional context: {emotional_influence}")
        
        # Message-specific reflection
        if len(message) > 100:
            reflection_parts.append("requiring detailed consideration and response")
        else:
            reflection_parts.append("allowing for focused and direct response")
        
        reflection = "Contextual reflection: " + "; ".join(reflection_parts) + "."
        
        logger.debug(f"🌐 Generated contextual reflection")
        return reflection
        
    except Exception as e:
        logger.error(f"❌ Error generating contextual reflection: {e}")
        return f"Reflection on message in context: {message[:50]}..."

def extract_context_themes(context: list[str]) -> list[str]:
    """Extract recurring themes from context"""
    try:
        if not context:
            return []
        
        # Simple keyword extraction from context
        word_frequency = {}
        
        for ctx in context:
            # Extract meaningful words (length > 4, alphabetic)
            words = [word.lower() for word in ctx.split() 
                    if len(word) > 4 and word.isalpha()]
            
            for word in words:
                word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Get top themes (words appearing more than once)
        themes = [word for word, count in word_frequency.items() 
                 if count > 1 and word not in ['user', 'system', 'response', 'message']]
        
        # Sort by frequency and return top themes
        themes.sort(key=lambda w: word_frequency[w], reverse=True)
        
        return themes[:5]  # Return top 5 themes
        
    except Exception as e:
        logger.error(f"❌ Error extracting context themes: {e}")
        return []

async def batch_reflection_processing(reflection_requests: List[dict]):
    """Process multiple reflection requests in batch"""
    try:
        logger.debug(f"📦 Processing {len(reflection_requests)} reflection requests")
        
        # Process reflections concurrently
        tasks = []
        for request in reflection_requests:
            task = generate_reflection(
                request["node_id"],
                request["message"],
                request.get("context", []),
                request.get("channel", "general")
            )
            tasks.append(task)
        
        # Execute all reflection tasks
        reflections = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_reflections = 0
        for i, reflection in enumerate(reflections):
            if not isinstance(reflection, Exception):
                successful_reflections += 1
            else:
                logger.error(f"❌ Reflection {i} failed: {reflection}")
        
        logger.info(f"✅ Processed {successful_reflections}/{len(reflection_requests)} reflections")
        return reflections
        
    except Exception as e:
        logger.error(f"❌ Error in batch reflection processing: {e}")
        return [] 