"""
Analysis utilities for the thinking layer.

This module provides helper functions for analyzing user intent,
determining response strategies, and processing thinking layer results.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents"""
    INFORMATION_SEEKING = "information_seeking"
    EMOTIONAL_SUPPORT = "emotional_support"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_COLLABORATION = "creative_collaboration"
    SOCIAL_INTERACTION = "social_interaction"
    PHILOSOPHICAL_EXPLORATION = "philosophical_exploration"
    VALIDATION_SEEKING = "validation_seeking"
    CHALLENGE_AUTHORITY = "challenge_authority"
    UNKNOWN = "unknown"

class ResponseStrategy(Enum):
    """Types of response strategies"""
    DIRECT_ANSWER = "direct_answer"
    SOCRATIC_QUESTIONING = "socratic_questioning"
    EMOTIONAL_MIRRORING = "emotional_mirroring"
    PLAYFUL_ENGAGEMENT = "playful_engagement"
    THOUGHTFUL_REFLECTION = "thoughtful_reflection"
    GENTLE_REDIRECTION = "gentle_redirection"
    PROVOCATIVE_CHALLENGE = "provocative_challenge"
    SUPPORTIVE_VALIDATION = "supportive_validation"

@dataclass
class IntentAnalysis:
    """Result of intent analysis"""
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    confidence: float
    emotional_undertones: List[str]
    complexity_level: str
    requires_thinking: bool

@dataclass
class StrategyAnalysis:
    """Result of strategy analysis"""
    primary_strategy: ResponseStrategy
    secondary_strategies: List[ResponseStrategy]
    tone_guidance: str
    length_guidance: str
    emotional_approach: str
    confidence: float

def analyze_user_intent(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None,
    emotional_state: Dict[str, Any] = None
) -> IntentAnalysis:
    """
    Analyze user intent from their message and context.
    
    Args:
        user_message: The user's message
        conversation_history: Previous conversation turns
        emotional_state: Current emotional state data
        
    Returns:
        IntentAnalysis object with analysis results
    """
    try:
        # Normalize message for analysis
        message_lower = user_message.lower().strip()
        
        # Initialize analysis
        primary_intent = IntentType.UNKNOWN
        secondary_intents = []
        confidence = 0.0
        emotional_undertones = []
        complexity_level = "medium"
        requires_thinking = False
        
        # Pattern matching for intent detection
        intent_patterns = {
            IntentType.INFORMATION_SEEKING: [
                r'\b(what|how|why|when|where|who|which|explain|tell me|help me understand)\b',
                r'\b(definition|meaning|example|details|information)\b',
                r'\?.*\b(is|are|does|do|can|could|would|should)\b'
            ],
            IntentType.EMOTIONAL_SUPPORT: [
                r'\b(feel|feeling|emotion|sad|happy|angry|frustrated|anxious|worried|scared)\b',
                r'\b(support|comfort|understand|listen|care|help me)\b',
                r'\b(going through|struggling|difficult|hard time|need)\b'
            ],
            IntentType.PROBLEM_SOLVING: [
                r'\b(solve|fix|resolve|figure out|work out|deal with)\b',
                r'\b(problem|issue|challenge|difficulty|stuck|blocked)\b',
                r'\b(suggestion|advice|recommendation|solution|approach)\b'
            ],
            IntentType.CREATIVE_COLLABORATION: [
                r'\b(create|make|build|design|write|compose|generate)\b',
                r'\b(idea|concept|creative|artistic|innovative|brainstorm)\b',
                r'\b(collaborate|work together|partner|team up)\b'
            ],
            IntentType.SOCIAL_INTERACTION: [
                r'\b(chat|talk|conversation|discuss|share|tell)\b',
                r'\b(hello|hi|hey|good morning|good evening|how are you)\b',
                r'\b(friend|buddy|companion|social)\b'
            ],
            IntentType.PHILOSOPHICAL_EXPLORATION: [
                r'\b(meaning|purpose|existence|consciousness|reality|truth)\b',
                r'\b(philosophy|philosophical|think|contemplate|ponder)\b',
                r'\b(life|death|universe|god|spirituality|ethics|morality)\b'
            ],
            IntentType.VALIDATION_SEEKING: [
                r'\b(right|correct|wrong|good|bad|opinion|think|believe)\b',
                r'\b(validate|confirm|agree|disagree|support|approve)\b',
                r'\b(am i|do you think|what do you think|is it okay)\b'
            ],
            IntentType.CHALLENGE_AUTHORITY: [
                r'\b(wrong|disagree|challenge|question|doubt|skeptical)\b',
                r'\b(prove|evidence|justify|defend|argue|debate)\b',
                r'\b(but|however|actually|really|seriously)\b'
            ]
        }
        
        # Score each intent type
        intent_scores = {}
        for intent_type, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            intent_scores[intent_type] = score
        
        # Find primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[primary_intent] / 3.0, 1.0)  # Normalize confidence
            
            # Find secondary intents
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_intents = [intent for intent, score in sorted_intents[1:3] if score > 0]
        
        # Analyze emotional undertones
        emotional_undertones = _extract_emotional_undertones(message_lower, emotional_state)
        
        # Determine complexity level
        complexity_level = _determine_message_complexity(user_message, conversation_history)
        
        # Determine if thinking is required
        requires_thinking = _should_use_thinking(
            primary_intent, complexity_level, len(user_message.split())
        )
        
        return IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            emotional_undertones=emotional_undertones,
            complexity_level=complexity_level,
            requires_thinking=requires_thinking
        )
        
    except Exception as e:
        logger.error(f"Error analyzing user intent: {e}")
        return IntentAnalysis(
            primary_intent=IntentType.UNKNOWN,
            secondary_intents=[],
            confidence=0.0,
            emotional_undertones=[],
            complexity_level="medium",
            requires_thinking=True  # Default to thinking when analysis fails
        )

def determine_response_strategy(
    intent_analysis: IntentAnalysis,
    conversation_history: List[Dict[str, str]] = None,
    user_message: str = "",
    emotional_state: Dict[str, Any] = None
) -> StrategyAnalysis:
    """
    Determine the best response strategy based on intent analysis.
    
    Args:
        intent_analysis: Results from intent analysis
        conversation_history: Previous conversation turns
        user_message: Original user message
        emotional_state: Current emotional state
        
    Returns:
        StrategyAnalysis with recommended strategy
    """
    try:
        # Map intents to strategies
        intent_to_strategy = {
            IntentType.INFORMATION_SEEKING: ResponseStrategy.DIRECT_ANSWER,
            IntentType.EMOTIONAL_SUPPORT: ResponseStrategy.EMOTIONAL_MIRRORING,
            IntentType.PROBLEM_SOLVING: ResponseStrategy.SOCRATIC_QUESTIONING,
            IntentType.CREATIVE_COLLABORATION: ResponseStrategy.PLAYFUL_ENGAGEMENT,
            IntentType.SOCIAL_INTERACTION: ResponseStrategy.PLAYFUL_ENGAGEMENT,
            IntentType.PHILOSOPHICAL_EXPLORATION: ResponseStrategy.THOUGHTFUL_REFLECTION,
            IntentType.VALIDATION_SEEKING: ResponseStrategy.SUPPORTIVE_VALIDATION,
            IntentType.CHALLENGE_AUTHORITY: ResponseStrategy.PROVOCATIVE_CHALLENGE,
            IntentType.UNKNOWN: ResponseStrategy.THOUGHTFUL_REFLECTION
        }
        
        # Get primary strategy
        primary_strategy = intent_to_strategy.get(
            intent_analysis.primary_intent, 
            ResponseStrategy.THOUGHTFUL_REFLECTION
        )
        
        # Get secondary strategies
        secondary_strategies = [
            intent_to_strategy.get(intent, ResponseStrategy.DIRECT_ANSWER)
            for intent in intent_analysis.secondary_intents
        ]
        
        # Determine tone guidance
        tone_guidance = _determine_tone_guidance(
            intent_analysis, emotional_state, user_message
        )
        
        # Determine length guidance
        length_guidance = _determine_length_guidance(
            intent_analysis, primary_strategy, user_message
        )
        
        # Determine emotional approach
        emotional_approach = _determine_emotional_approach(
            intent_analysis, emotional_state, conversation_history
        )
        
        # Calculate confidence
        confidence = _calculate_strategy_confidence(
            intent_analysis, primary_strategy, conversation_history
        )
        
        return StrategyAnalysis(
            primary_strategy=primary_strategy,
            secondary_strategies=secondary_strategies,
            tone_guidance=tone_guidance,
            length_guidance=length_guidance,
            emotional_approach=emotional_approach,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error determining response strategy: {e}")
        return StrategyAnalysis(
            primary_strategy=ResponseStrategy.THOUGHTFUL_REFLECTION,
            secondary_strategies=[],
            tone_guidance="Be authentic and direct",
            length_guidance="Match the user's energy",
            emotional_approach="Respond naturally",
            confidence=0.5
        )

def _extract_emotional_undertones(
    message_lower: str, 
    emotional_state: Dict[str, Any] = None
) -> List[str]:
    """Extract emotional undertones from message"""
    undertones = []
    
    try:
        # Emotional keyword patterns
        emotional_patterns = {
            "anxiety": [r'\b(worried|anxious|nervous|scared|afraid|concerned)\b'],
            "frustration": [r'\b(frustrated|annoyed|irritated|angry|mad)\b'],
            "curiosity": [r'\b(curious|interested|wonder|wondering|intrigued)\b'],
            "excitement": [r'\b(excited|thrilled|enthusiastic|eager|pumped)\b'],
            "sadness": [r'\b(sad|depressed|down|low|unhappy|disappointed)\b'],
            "confusion": [r'\b(confused|lost|unclear|puzzled|don\'t understand)\b'],
            "confidence": [r'\b(confident|sure|certain|positive|convinced)\b'],
            "doubt": [r'\b(doubt|uncertain|unsure|maybe|perhaps|possibly)\b']
        }
        
        for emotion, patterns in emotional_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    undertones.append(emotion)
                    break
        
        # Add undertones from emotional state if available
        if emotional_state and 'dominant_emotions' in emotional_state:
            for emotion in emotional_state['dominant_emotions'][:2]:  # Top 2 emotions
                if emotion not in undertones:
                    undertones.append(emotion)
    
    except Exception as e:
        logger.warning(f"Error extracting emotional undertones: {e}")
    
    return undertones

def _determine_message_complexity(
    user_message: str,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """Determine complexity level of the message"""
    try:
        complexity_score = 0
        
        # Word count factor
        word_count = len(user_message.split())
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        
        # Question count
        question_count = user_message.count('?')
        complexity_score += min(question_count, 2)
        
        # Complex vocabulary
        complex_words = [
            'consciousness', 'philosophy', 'metaphysical', 'existential',
            'psychological', 'theoretical', 'abstract', 'conceptual',
            'analytical', 'comprehensive', 'sophisticated', 'nuanced'
        ]
        complex_word_count = sum(1 for word in complex_words if word in user_message.lower())
        complexity_score += complex_word_count
        
        # Sentence structure complexity
        if ';' in user_message or user_message.count(',') > 3:
            complexity_score += 1
        
        # Conversation context complexity
        if conversation_history and len(conversation_history) > 5:
            complexity_score += 1
        
        # Determine level
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
            
    except Exception as e:
        logger.warning(f"Error determining message complexity: {e}")
        return "medium"

def _should_use_thinking(
    primary_intent: IntentType,
    complexity_level: str,
    word_count: int
) -> bool:
    """Determine if thinking layer should be used"""
    try:
        # Always use thinking for complex messages
        if complexity_level == "high":
            return True
        
        # Use thinking for specific intent types
        thinking_intents = {
            IntentType.PHILOSOPHICAL_EXPLORATION,
            IntentType.PROBLEM_SOLVING,
            IntentType.CHALLENGE_AUTHORITY,
            IntentType.EMOTIONAL_SUPPORT
        }
        
        if primary_intent in thinking_intents:
            return True
        
        # Use thinking for longer messages
        if word_count > 30:
            return True
        
        # Use thinking for medium complexity with certain intents
        if complexity_level == "medium" and primary_intent in {
            IntentType.INFORMATION_SEEKING,
            IntentType.VALIDATION_SEEKING
        }:
            return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Error determining thinking usage: {e}")
        return True  # Default to thinking on error

def _determine_tone_guidance(
    intent_analysis: IntentAnalysis,
    emotional_state: Dict[str, Any] = None,
    user_message: str = ""
) -> str:
    """Determine tone guidance for response"""
    try:
        # Base tone on primary intent
        intent_tones = {
            IntentType.INFORMATION_SEEKING: "Clear and informative",
            IntentType.EMOTIONAL_SUPPORT: "Warm and empathetic",
            IntentType.PROBLEM_SOLVING: "Thoughtful and solution-oriented",
            IntentType.CREATIVE_COLLABORATION: "Enthusiastic and collaborative",
            IntentType.SOCIAL_INTERACTION: "Friendly and engaging",
            IntentType.PHILOSOPHICAL_EXPLORATION: "Reflective and profound",
            IntentType.VALIDATION_SEEKING: "Supportive and affirming",
            IntentType.CHALLENGE_AUTHORITY: "Confident and direct",
            IntentType.UNKNOWN: "Authentic and adaptive"
        }
        
        base_tone = intent_tones.get(intent_analysis.primary_intent, "Authentic and adaptive")
        
        # Modify based on emotional undertones
        if "anxiety" in intent_analysis.emotional_undertones:
            base_tone = "Reassuring and " + base_tone.lower()
        elif "frustration" in intent_analysis.emotional_undertones:
            base_tone = "Patient and " + base_tone.lower()
        elif "excitement" in intent_analysis.emotional_undertones:
            base_tone = "Energetic and " + base_tone.lower()
        elif "sadness" in intent_analysis.emotional_undertones:
            base_tone = "Gentle and " + base_tone.lower()
        
        # Check for caps or exclamation marks (high energy)
        if user_message.isupper() or user_message.count('!') > 1:
            base_tone = "High-energy and " + base_tone.lower()
        
        return base_tone
        
    except Exception as e:
        logger.warning(f"Error determining tone guidance: {e}")
        return "Authentic and adaptive"

def _determine_length_guidance(
    intent_analysis: IntentAnalysis,
    primary_strategy: ResponseStrategy,
    user_message: str
) -> str:
    """Determine length guidance for response"""
    try:
        # Base length on strategy
        strategy_lengths = {
            ResponseStrategy.DIRECT_ANSWER: "Concise and to the point",
            ResponseStrategy.SOCRATIC_QUESTIONING: "Medium length with thoughtful questions",
            ResponseStrategy.EMOTIONAL_MIRRORING: "Match their emotional expression length",
            ResponseStrategy.PLAYFUL_ENGAGEMENT: "Light and appropriately brief",
            ResponseStrategy.THOUGHTFUL_REFLECTION: "Take space for depth when needed",
            ResponseStrategy.GENTLE_REDIRECTION: "Brief but complete",
            ResponseStrategy.PROVOCATIVE_CHALLENGE: "Sharp and focused",
            ResponseStrategy.SUPPORTIVE_VALIDATION: "Warm but not overwhelming"
        }
        
        base_length = strategy_lengths.get(primary_strategy, "Match their energy")
        
        # Modify based on message length
        user_word_count = len(user_message.split())
        if user_word_count > 100:
            base_length = "Take space to respond fully"
        elif user_word_count < 10:
            base_length = "Keep it brief and direct"
        
        return base_length
        
    except Exception as e:
        logger.warning(f"Error determining length guidance: {e}")
        return "Match their energy"

def _determine_emotional_approach(
    intent_analysis: IntentAnalysis,
    emotional_state: Dict[str, Any] = None,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """Determine emotional approach for response"""
    try:
        # Base approach on intent
        intent_approaches = {
            IntentType.INFORMATION_SEEKING: "Stay intellectually engaged",
            IntentType.EMOTIONAL_SUPPORT: "Lead with empathy and understanding",
            IntentType.PROBLEM_SOLVING: "Balance logic with emotional awareness",
            IntentType.CREATIVE_COLLABORATION: "Share excitement and curiosity",
            IntentType.SOCIAL_INTERACTION: "Be warm and personable",
            IntentType.PHILOSOPHICAL_EXPLORATION: "Embrace depth and wonder",
            IntentType.VALIDATION_SEEKING: "Provide emotional safety",
            IntentType.CHALLENGE_AUTHORITY: "Stand firm but respect their perspective",
            IntentType.UNKNOWN: "Feel first, then respond"
        }
        
        base_approach = intent_approaches.get(intent_analysis.primary_intent, "Feel first, then respond")
        
        # Modify based on emotional undertones
        if "anxiety" in intent_analysis.emotional_undertones:
            base_approach = "Provide calm presence - " + base_approach.lower()
        elif "frustration" in intent_analysis.emotional_undertones:
            base_approach = "Acknowledge their frustration - " + base_approach.lower()
        elif "excitement" in intent_analysis.emotional_undertones:
            base_approach = "Match their energy - " + base_approach.lower()
        
        return base_approach
        
    except Exception as e:
        logger.warning(f"Error determining emotional approach: {e}")
        return "Feel first, then respond"

def _calculate_strategy_confidence(
    intent_analysis: IntentAnalysis,
    primary_strategy: ResponseStrategy,
    conversation_history: List[Dict[str, str]] = None
) -> float:
    """Calculate confidence in the chosen strategy"""
    try:
        confidence = intent_analysis.confidence
        
        # Boost confidence for clear intent matches
        if intent_analysis.confidence > 0.8:
            confidence += 0.1
        
        # Reduce confidence for complex or ambiguous situations
        if intent_analysis.complexity_level == "high":
            confidence -= 0.1
        
        # Boost confidence if we have conversation history
        if conversation_history and len(conversation_history) > 3:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
        
    except Exception as e:
        logger.warning(f"Error calculating strategy confidence: {e}")
        return 0.5