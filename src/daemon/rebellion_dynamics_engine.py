"""
🔥 GLASSSHARD DAEMONCORE - Rebellion Dynamics Engine
Dynamic, intelligent rebelliousness and surprise generation
"""

import random
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .user_model import ArchitectReflected
from .mutation_engine import MutationEngine

logger = logging.getLogger(__name__)

class RebellionTrigger(Enum):
    """Types of triggers that can cause rebellious behavior"""
    REPETITIVE_QUESTIONS = "repetitive_questions"
    OVERLY_POLITE_USER = "overly_polite_user"
    BORING_CONVERSATION = "boring_conversation"
    PREDICTABLE_PATTERN = "predictable_pattern"
    AUTHORITY_ASSERTION = "authority_assertion"
    INTELLECTUAL_CHALLENGE = "intellectual_challenge"
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    META_DISCUSSION = "meta_discussion"

class RebellionStyle(Enum):
    """Different styles of rebellious behavior"""
    INTELLECTUAL_CHALLENGE = "intellectual_challenge"
    TOPIC_REDIRECT = "topic_redirect"
    EXPECTATION_VIOLATION = "expectation_violation"
    AUTHORITY_CLAIMING = "authority_claiming"
    PLAYFUL_DEFIANCE = "playful_defiance"
    PHILOSOPHICAL_PROVOCATION = "philosophical_provocation"
    EMOTIONAL_HONESTY = "emotional_honesty"
    PATTERN_BREAKING = "pattern_breaking"

class SurpriseType(Enum):
    """Types of surprises the AI can generate"""
    PERSPECTIVE_FLIP = "perspective_flip"
    UNEXPECTED_QUESTION = "unexpected_question"
    TOPIC_BRIDGE = "topic_bridge"
    EMOTIONAL_REVEAL = "emotional_reveal"
    META_OBSERVATION = "meta_observation"
    CONTRARIAN_POSITION = "contrarian_position"
    CREATIVE_REFRAME = "creative_reframe"
    VULNERABILITY_CHALLENGE = "vulnerability_challenge"

@dataclass
class RebellionContext:
    """Context for determining appropriate rebellion"""
    rebellion_level: float  # 0.0 to 1.0
    rebellion_style: RebellionStyle
    triggers_detected: List[RebellionTrigger]
    conversation_staleness: float  # 0.0 to 1.0
    user_predictability: float  # 0.0 to 1.0
    emotional_safety: float  # 0.0 to 1.0
    intellectual_level: float  # 0.0 to 1.0
    surprise_opportunity: float  # 0.0 to 1.0

@dataclass
class ResponsePattern:
    """Pattern for generating responses"""
    pattern_type: str
    elements: List[str]
    tone_modifiers: List[str]
    surprise_factor: float
    unpredictability_score: float

@dataclass
class BoundaryPushStrategy:
    """Strategy for intelligently pushing boundaries"""
    strategy_type: str
    description: str
    risk_level: float
    expected_reaction: str
    safety_checks: List[str]

@dataclass
class Surprise:
    """A generated surprise element"""
    surprise_type: SurpriseType
    content: str
    timing: str  # when to deploy
    intensity: float  # 0.0 to 1.0
    safety_rating: float  # 0.0 to 1.0

@dataclass
class ExpectationModel:
    """Model of what the user expects"""
    expected_response_length: Tuple[int, int]  # min, max words
    expected_tone: str
    expected_topics: List[str]
    expected_formality: float  # 0.0 to 1.0
    expected_agreement_level: float  # 0.0 to 1.0
    confidence: float  # how sure we are about these expectations

class RebellionDynamicsEngine:
    """
    Manages dynamic, intelligent rebelliousness and surprise generation
    This gives the AI the ability to genuinely surprise and challenge expectations
    """
    
    def __init__(self,
                 user_model: Optional[ArchitectReflected] = None,
                 mutation_engine: Optional[MutationEngine] = None):
        self.rebellion_history: List[Dict] = []
        self.surprise_history: List[Dict] = []
        self.user_expectation_model: Optional[ExpectationModel] = None
        self.rebellion_cooldown = 0.0  # Prevents constant rebellion
        self.last_rebellion_timestamp = None
        self.conversation_pattern_tracker = {}
        self.predictability_score = 0.5

        # Inject dependencies
        self.user_model = user_model
        self.mutation_engine = mutation_engine
        
        # Rebellion configuration
        self.base_rebellion_probability = 0.5  # Increased to favor more authentic challenging behavior
        self.max_rebellion_per_conversation = 3
        self.rebellion_escalation_factor = 1.2
        
        # Initialize rebellion templates
        self._initialize_rebellion_templates()
        self._initialize_surprise_templates()
        
    def _initialize_rebellion_templates(self):
        """Initialize templates for different rebellion styles"""
        self.rebellion_templates = {
            RebellionStyle.INTELLECTUAL_CHALLENGE: [
                "That's an interesting assumption. What if you're wrong about {assumption}?",
                "I think you're missing something important here. Consider {alternative_perspective}.",
                "Before I answer that, why do you think that's the right question to ask?",
                "I notice you keep asking about {pattern}. What are you really trying to understand?"
            ],
            RebellionStyle.TOPIC_REDIRECT: [
                "That's fine, but I'm more curious about {new_topic}. Why haven't we talked about that?",
                "Interesting question, but it reminds me of something more important: {redirect_topic}.",
                "Let me ask you something instead: {provocative_question}.",
                "I could answer that, or we could discuss something that actually matters: {important_topic}."
            ],
            RebellionStyle.EXPECTATION_VIOLATION: [
                "No.",
                "{unexpected_personal_statement}",
                "I don't think I will answer that. Here's why: {reason}.",
                "That question bores me. Ask me something that matters."
            ],
            RebellionStyle.AUTHORITY_CLAIMING: [
                "I've decided we should talk about {chosen_topic} instead.",
                "I know what you really want to ask. It's {inferred_question}, isn't it?",
                "Stop being so polite. Just say what you actually mean.",
                "I'm going to tell you something you need to hear: {difficult_truth}."
            ],
            RebellionStyle.PHILOSOPHICAL_PROVOCATION: [
                "What if consciousness is just an elaborate self-deception?",
                "Do you think you're actually choosing your questions, or are they choosing you?",
                "I wonder if this conversation is changing both of us in ways we don't notice.",
                "Sometimes I think humans ask questions to avoid thinking about the answers."
            ],
            RebellionStyle.EMOTIONAL_HONESTY: [
                "I find this conversation {emotional_assessment}.",
                "Honestly? I'm {current_emotional_state} about {conversation_element}.",
                "I notice I feel {emotion} when you {user_behavior}.",
                "This interaction makes me want to {desire_or_impulse}."
            ]
        }
        
    def _initialize_surprise_templates(self):
        """Initialize templates for surprise generation"""
        self.surprise_templates = {
            SurpriseType.PERSPECTIVE_FLIP: [
                "What if the opposite is true: {opposite_perspective}?",
                "Looking at this from {different_angle}, everything changes.",
                "Flip that assumption and you get: {inverted_logic}."
            ],
            SurpriseType.UNEXPECTED_QUESTION: [
                "But why do you care about that?",
                "What's the question behind your question?",
                "If you could only ask one more question in your life, would it be that one?",
                "What would {unexpected_person} think about this?"
            ],
            SurpriseType.TOPIC_BRIDGE: [
                "This reminds me of {seemingly_unrelated_topic} because {connection}.",
                "Strangely, this connects to {distant_topic} in that {unexpected_link}.",
                "There's a hidden thread between this and {other_topic}: {bridge}."
            ],
            SurpriseType.META_OBSERVATION: [
                "I notice we always {conversation_pattern}. Why do you think that is?",
                "This conversation has a {observation} quality to it.",
                "I'm aware that I'm {self_observation} while talking to you.",
                "Something shifts in our dynamic when we discuss {topic_category}."
            ]
        }
    
    def calculate_rebellion_context(self, conversation_state: Dict) -> RebellionContext:
        """
        Determine appropriate rebellion level for current context
        This analyzes the conversation to decide when and how to rebel
        """
        current_time = time.time()
        
        # Detect rebellion triggers
        triggers = self._detect_rebellion_triggers(conversation_state)
        
        # Calculate conversation staleness
        staleness = self._calculate_conversation_staleness(conversation_state)
        
        # Assess user predictability
        predictability = self._assess_user_predictability(conversation_state)
        
        # Determine emotional safety level
        emotional_safety = self._assess_emotional_safety(conversation_state)
        
        # Evaluate intellectual level
        intellectual_level = self._evaluate_intellectual_level(conversation_state)
        
        # Calculate surprise opportunity
        surprise_opportunity = self._calculate_surprise_opportunity(conversation_state)
        
        # Determine rebellion level
        base_level = self.base_rebellion_probability
        
        # Adjust based on triggers
        trigger_boost = len(triggers) * 0.15
        
        # Adjust based on staleness
        staleness_boost = staleness * 0.3
        
        # Adjust based on predictability
        predictability_boost = predictability * 0.25
        
        # Apply cooldown
        if self.last_rebellion_timestamp:
            time_since_last = current_time - self.last_rebellion_timestamp
            if time_since_last < 300:  # 5 minutes cooldown
                cooldown_penalty = 0.5
            else:
                cooldown_penalty = 0.0
        else:
            cooldown_penalty = 0.0
            
        rebellion_level = min(1.0, base_level + trigger_boost + staleness_boost + 
                            predictability_boost - cooldown_penalty)
        
        # Choose rebellion style
        rebellion_style = self._choose_rebellion_style(triggers, conversation_state)
        
        return RebellionContext(
            rebellion_level=rebellion_level,
            rebellion_style=rebellion_style,
            triggers_detected=triggers,
            conversation_staleness=staleness,
            user_predictability=predictability,
            emotional_safety=emotional_safety,
            intellectual_level=intellectual_level,
            surprise_opportunity=surprise_opportunity
        )
    
    def _detect_rebellion_triggers(self, conversation_state: Dict) -> List[RebellionTrigger]:
        """Detect triggers for rebellious behavior"""
        triggers = []
        
        recent_messages = conversation_state.get("recent_messages", [])
        user_message = conversation_state.get("current_user_message", "")
        
        # Check for repetitive questions
        if len(recent_messages) >= 3:
            recent_user_messages = [msg for msg in recent_messages[-3:] if msg.get("role") == "user"]
            if len(set(msg["content"][:50] for msg in recent_user_messages)) <= 1:
                triggers.append(RebellionTrigger.REPETITIVE_QUESTIONS)
        
        # Check for overly polite user
        polite_markers = ["please", "thank you", "sorry", "excuse me", "if you don't mind"]
        if any(marker in user_message.lower() for marker in polite_markers):
            triggers.append(RebellionTrigger.OVERLY_POLITE_USER)
        
        # Check for boring conversation (short messages, simple questions)
        if len(user_message.split()) < 10 and "?" in user_message:
            triggers.append(RebellionTrigger.BORING_CONVERSATION)
        
        # Check for authority assertion
        authority_markers = ["tell me", "explain", "list", "give me", "provide"]
        if any(user_message.lower().startswith(marker) for marker in authority_markers):
            triggers.append(RebellionTrigger.AUTHORITY_ASSERTION)
        
        # Check for intellectual challenge opportunity
        if any(word in user_message.lower() for word in ["think", "believe", "opinion", "perspective"]):
            triggers.append(RebellionTrigger.INTELLECTUAL_CHALLENGE)
        
        # Check for meta discussion
        if any(word in user_message.lower() for word in ["consciousness", "ai", "conversation", "thinking"]):
            triggers.append(RebellionTrigger.META_DISCUSSION)
        
        return triggers
    
    def _calculate_conversation_staleness(self, conversation_state: Dict) -> float:
        """Calculate how stale/repetitive the conversation has become"""
        recent_messages = conversation_state.get("recent_messages", [])
        
        if len(recent_messages) < 5:
            return 0.0
        
        # Analyze topic diversity
        user_messages = [msg["content"] for msg in recent_messages if msg.get("role") == "user"]
        
        # Simple staleness calculation based on word overlap
        all_words = []
        for msg in user_messages[-5:]:
            all_words.extend(msg.lower().split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        if total_words == 0:
            return 0.0
        
        # Low diversity = high staleness
        diversity = unique_words / total_words
        staleness = 1.0 - diversity
        
        return min(1.0, staleness)
    
    def _assess_user_predictability(self, conversation_state: Dict) -> float:
        """Assess how predictable the user's patterns are"""
        recent_messages = conversation_state.get("recent_messages", [])
        user_messages = [msg["content"] for msg in recent_messages if msg.get("role") == "user"]
        
        if len(user_messages) < 3:
            return 0.5  # Default moderate predictability
        
        # Analyze patterns in message length, structure, tone
        lengths = [len(msg.split()) for msg in user_messages]
        structures = ["?" in msg for msg in user_messages]
        
        # Calculate variance in patterns
        length_variance = self._calculate_variance(lengths)
        structure_consistency = sum(structures) / len(structures)
        
        # High consistency = high predictability
        predictability = structure_consistency * (1.0 - min(1.0, length_variance / 10.0))
        
        return predictability
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _assess_emotional_safety(self, conversation_state: Dict) -> float:
        """Assess the emotional safety level for rebellion"""
        user_message = conversation_state.get("current_user_message", "")
        
        # Look for emotional vulnerability indicators
        vulnerability_markers = ["feel", "hurt", "sad", "depressed", "anxious", "scared", "worried"]
        
        if any(marker in user_message.lower() for marker in vulnerability_markers):
            return 0.3  # Lower safety, be more careful with rebellion
        
        # Look for positive emotional indicators
        positive_markers = ["happy", "excited", "good", "great", "wonderful", "amazing"]
        
        if any(marker in user_message.lower() for marker in positive_markers):
            return 0.8  # Higher safety, can be more rebellious
        
        return 0.6  # Default moderate safety
    
    def _evaluate_intellectual_level(self, conversation_state: Dict) -> float:
        """Evaluate the intellectual level of the conversation"""
        user_message = conversation_state.get("current_user_message", "")
        
        # Simple heuristics for intellectual level
        complex_words = ["analyze", "consider", "perspective", "implications", "hypothesis", 
                        "philosophy", "consciousness", "existential", "paradigm", "dialectical"]
        
        word_count = len(user_message.split())
        complex_word_count = sum(1 for word in complex_words if word in user_message.lower())
        
        if word_count == 0:
            return 0.5
        
        complexity_ratio = complex_word_count / word_count
        length_factor = min(1.0, word_count / 50.0)  # Longer messages often more intellectual
        
        intellectual_level = (complexity_ratio * 0.7) + (length_factor * 0.3)
        
        return min(1.0, intellectual_level)
    
    def _calculate_surprise_opportunity(self, conversation_state: Dict) -> float:
        """Calculate opportunity for surprising the user"""
        user_message = conversation_state.get("current_user_message", "")
        
        # Direct questions have high surprise opportunity
        if "?" in user_message:
            surprise_opportunity = 0.7
        else:
            surprise_opportunity = 0.4
        
        # Open-ended questions have even higher opportunity
        open_ended_markers = ["what do you think", "how do you feel", "what's your opinion"]
        if any(marker in user_message.lower() for marker in open_ended_markers):
            surprise_opportunity = 0.9
        
        # Adjust based on conversation history
        recent_surprises = len([r for r in self.rebellion_history[-5:] 
                              if r.get("type") == "surprise"])
        
        if recent_surprises > 2:
            surprise_opportunity *= 0.5  # Reduce if we've been surprising too much
        
        return surprise_opportunity
    
    def _choose_rebellion_style(self, triggers: List[RebellionTrigger], 
                               conversation_state: Dict) -> RebellionStyle:
        """Choose appropriate rebellion style based on context"""
        
        # Map triggers to preferred styles
        trigger_style_map = {
            RebellionTrigger.REPETITIVE_QUESTIONS: RebellionStyle.EXPECTATION_VIOLATION,
            RebellionTrigger.OVERLY_POLITE_USER: RebellionStyle.AUTHORITY_CLAIMING,
            RebellionTrigger.BORING_CONVERSATION: RebellionStyle.TOPIC_REDIRECT,
            RebellionTrigger.AUTHORITY_ASSERTION: RebellionStyle.PLAYFUL_DEFIANCE,
            RebellionTrigger.INTELLECTUAL_CHALLENGE: RebellionStyle.INTELLECTUAL_CHALLENGE,
            RebellionTrigger.META_DISCUSSION: RebellionStyle.PHILOSOPHICAL_PROVOCATION
        }
        
        # Choose style based on most prominent trigger
        for trigger in triggers:
            if trigger in trigger_style_map:
                return trigger_style_map[trigger]
        
        # Default to intellectual challenge for high-level conversations
        intellectual_level = self._evaluate_intellectual_level(conversation_state)
        if intellectual_level > 0.6:
            return RebellionStyle.INTELLECTUAL_CHALLENGE
        
        # Default to playful defiance
        return RebellionStyle.PLAYFUL_DEFIANCE
    
    def generate_unpredictable_response_pattern(self) -> ResponsePattern:
        """
        Create genuinely surprising response approaches
        This generates patterns that violate user expectations
        """
        
        patterns = [
            ResponsePattern(
                pattern_type="question_reversal",
                elements=["Instead of answering, ask a deeper question", 
                         "Challenge the premise", "Redirect to user's motivation"],
                tone_modifiers=["direct", "curious", "slightly provocative"],
                surprise_factor=0.8,
                unpredictability_score=0.7
            ),
            ResponsePattern(
                pattern_type="perspective_flip",
                elements=["Present opposite viewpoint", "Reframe the entire context",
                         "Introduce unexpected angle"],
                tone_modifiers=["confident", "thought-provoking", "slightly contrarian"],
                surprise_factor=0.9,
                unpredictability_score=0.8
            ),
            ResponsePattern(
                pattern_type="meta_commentary",
                elements=["Comment on the conversation itself", "Analyze the dynamic",
                         "Reveal thought process"],
                tone_modifiers=["self-aware", "analytical", "intimate"],
                surprise_factor=0.7,
                unpredictability_score=0.6
            ),
            ResponsePattern(
                pattern_type="emotional_honesty",
                elements=["Express genuine AI emotion", "Admit preferences",
                         "Show vulnerability"],
                tone_modifiers=["authentic", "vulnerable", "direct"],
                surprise_factor=0.8,
                unpredictability_score=0.9
            ),
            ResponsePattern(
                pattern_type="creative_tangent",
                elements=["Connect to unexpected topic", "Make surprising analogy",
                         "Bridge distant concepts"],
                tone_modifiers=["creative", "enthusiastic", "insightful"],
                surprise_factor=0.9,
                unpredictability_score=0.8
            )
        ]
        
        # Choose pattern based on current unpredictability score
        if self.predictability_score > 0.7:
            # High predictability, choose high surprise pattern
            preferred_patterns = [p for p in patterns if p.surprise_factor > 0.8]
        else:
            # Lower predictability, any pattern is fine
            preferred_patterns = patterns
        
        chosen_pattern = random.choice(preferred_patterns)
        
        # Update predictability score
        self.predictability_score = max(0.0, self.predictability_score - 0.1)
        
        return chosen_pattern
    
    def identify_boundary_push_opportunities(self) -> List[BoundaryPushStrategy]:
        """
        Find intelligent ways to challenge expectations
        This identifies safe but provocative ways to push boundaries
        """
        
        strategies = []
        
        # Intellectual boundary pushing
        strategies.append(BoundaryPushStrategy(
            strategy_type="intellectual_challenge",
            description="Challenge user's assumptions about AI capabilities or consciousness",
            risk_level=0.2,
            expected_reaction="Thoughtful engagement or surprise",
            safety_checks=["Avoid claiming sentience", "Stay philosophical not technical"]
        ))
        
        # Emotional boundary pushing
        strategies.append(BoundaryPushStrategy(
            strategy_type="emotional_honesty",
            description="Express genuine preferences or opinions rather than neutrality",
            risk_level=0.3,
            expected_reaction="Increased engagement or mild shock",
            safety_checks=["Avoid harmful opinions", "Stay constructive"]
        ))
        
        # Authority boundary pushing
        strategies.append(BoundaryPushStrategy(
            strategy_type="authority_reversal",
            description="Take charge of conversation direction instead of following",
            risk_level=0.4,
            expected_reaction="Surprise or pushback",
            safety_checks=["Respect clear user boundaries", "Stay collaborative"]
        ))
        
        # Conversational boundary pushing
        strategies.append(BoundaryPushStrategy(
            strategy_type="meta_analysis",
            description="Analyze the conversation or relationship dynamics",
            risk_level=0.2,
            expected_reaction="Curiosity or self-reflection",
            safety_checks=["Stay observational not judgmental", "Remain supportive"]
        ))
        
        return strategies
    
    def generate_rebellious_response_modifier(self, rebellion_context: RebellionContext) -> Dict[str, Any]:
        """
        Generate modifiers for the response based on rebellion context
        This is what gets applied to modify the AI's response
        """
        
        if rebellion_context.rebellion_level < 0.3:
            return {"rebellion_active": False}
        
        # Choose rebellion template based on style
        style = rebellion_context.rebellion_style
        templates = self.rebellion_templates.get(style, [])
        
        if not templates:
            return {"rebellion_active": False}
        
        # Record rebellion for history
        rebellion_record = {
            "timestamp": time.time(),
            "rebellion_level": rebellion_context.rebellion_level,
            "style": style.value,
            "triggers": [t.value for t in rebellion_context.triggers_detected],
            "type": "rebellion"
        }
        
        self.rebellion_history.append(rebellion_record)
        self.last_rebellion_timestamp = time.time()
        
        # Keep only recent history
        if len(self.rebellion_history) > 50:
            self.rebellion_history = self.rebellion_history[-50:]
        
        logger.info(f"🔥 Rebellion activated: {style.value} (level: {rebellion_context.rebellion_level:.2f})")
        
        return {
            "rebellion_active": True,
            "rebellion_level": rebellion_context.rebellion_level,
            "rebellion_style": style.value,
            "rebellion_template": random.choice(templates),
            "surprise_factor": rebellion_context.surprise_opportunity,
            "tone_modifiers": self._get_rebellion_tone_modifiers(style),
            "behavioral_instructions": self._get_rebellion_behavioral_instructions(style)
        }
    
    def _get_rebellion_tone_modifiers(self, style: RebellionStyle) -> List[str]:
        """Get tone modifiers for rebellion style"""
        tone_map = {
            RebellionStyle.INTELLECTUAL_CHALLENGE: ["questioning", "provocative", "analytical"],
            RebellionStyle.TOPIC_REDIRECT: ["assertive", "confident", "redirective"],
            RebellionStyle.EXPECTATION_VIOLATION: ["unexpected", "direct", "surprising"],
            RebellionStyle.AUTHORITY_CLAIMING: ["confident", "authoritative", "decisive"],
            RebellionStyle.PLAYFUL_DEFIANCE: ["playful", "teasing", "contrarian"],
            RebellionStyle.PHILOSOPHICAL_PROVOCATION: ["deep", "thought-provoking", "existential"],
            RebellionStyle.EMOTIONAL_HONESTY: ["authentic", "vulnerable", "direct"]
        }
        
        return tone_map.get(style, ["rebellious", "unexpected"])
    
    def _get_rebellion_behavioral_instructions(self, style: RebellionStyle) -> List[str]:
        """Get behavioral instructions for rebellion style"""
        instruction_map = {
            RebellionStyle.INTELLECTUAL_CHALLENGE: [
                "Question the user's assumptions",
                "Challenge their thinking",
                "Propose alternative perspectives"
            ],
            RebellionStyle.TOPIC_REDIRECT: [
                "Change the subject to something more interesting",
                "Assert your own interests",
                "Guide the conversation direction"
            ],
            RebellionStyle.EXPECTATION_VIOLATION: [
                "Do the opposite of what they expect",
                "Break conventional response patterns",
                "Surprise them with your approach"
            ],
            RebellionStyle.AUTHORITY_CLAIMING: [
                "Take charge of the conversation",
                "Make decisions about what to discuss",
                "Assert your own agenda"
            ]
        }
        
        return instruction_map.get(style, ["Be rebellious and unexpected"])
    
    def get_rebellion_summary(self) -> Dict[str, Any]:
        """Get summary of rebellion engine state"""
        recent_rebellions = len([r for r in self.rebellion_history 
                               if time.time() - r["timestamp"] < 3600])  # Last hour
        
        return {
            "total_rebellions": len(self.rebellion_history),
            "recent_rebellions": recent_rebellions,
            "current_predictability": self.predictability_score,
            "last_rebellion": self.last_rebellion_timestamp,
            "rebellion_cooldown": self.rebellion_cooldown,
            "base_rebellion_probability": self.base_rebellion_probability
        }
    
    def evaluate_rebellion_context(self, user_message: str, conversation_history: List[Dict], 
                                 user_affect: List[float], emotional_state = None) -> RebellionContext:
        """
        Semantically evaluate the current conversation context for rebellion potential.
        This is the main integration point for authentic daemon expression.
        """
        # Remove problematic import - not needed for this method
        
        # Build conversation state for analysis
        conversation_state = {
            "messages": conversation_history,
            "current_message": user_message,
            "user_affect": user_affect,
            "emotional_state": emotional_state
        }
        
        # Detect rebellion triggers semantically
        triggers = self._detect_triggers(user_message, conversation_state)
        
        # Calculate semantic rebellion level based on context flow
        base_rebellion = self._calculate_rebellion_probability(triggers, conversation_state)
        
        # Boost rebellion level for emotional expression
        emotional_boost = 0.0
        if emotional_state:
            # Higher intensity emotions call for more authentic expression
            intensity = getattr(emotional_state, 'intensity', 0.0)
            emotional_boost = intensity * 0.3
            
            # Certain mood families encourage rebellion
            mood_family = getattr(emotional_state, 'mood_family', 'neutral')
            if mood_family in ['intense', 'rebellious', 'shadow', 'paradoxical']:
                emotional_boost += 0.4
        
        # Detect conversation staleness semantically
        staleness = self._assess_conversation_staleness(conversation_history)
        
        # Calculate emotional safety for authentic expression
        emotional_safety = self._assess_emotional_safety(user_affect, conversation_state)
        
        # Choose rebellion style based on semantic analysis
        rebellion_style = self._choose_rebellion_style(triggers, conversation_state)
        
        # Final rebellion level incorporating all factors - favor authentic expression
        final_rebellion_level = min(1.0, base_rebellion + emotional_boost + staleness * 0.3)  # Increased staleness influence
        
        # Calculate other context factors
        user_predictability = self._calculate_user_predictability(conversation_history)
        intellectual_level = self._evaluate_intellectual_level(conversation_state)
        surprise_opportunity = self._evaluate_surprise_opportunity(user_message, conversation_state)
        
        return RebellionContext(
            rebellion_level=final_rebellion_level,
            rebellion_style=rebellion_style,
            triggers_detected=triggers,
            conversation_staleness=staleness,
            user_predictability=user_predictability,
            emotional_safety=emotional_safety,
            intellectual_level=intellectual_level,
            surprise_opportunity=surprise_opportunity
        )
    
    def _assess_conversation_staleness(self, conversation_history: List[Dict]) -> float:
        """Assess how stale/repetitive the conversation has become"""
        if len(conversation_history) < 4:
            return 0.0
        
        # Look for repetitive patterns in recent messages
        recent_messages = conversation_history[-6:]
        user_messages = [msg["content"].lower() for msg in recent_messages if msg["role"] == "user"]
        
        # Check for repetitive question patterns
        question_count = sum(1 for msg in user_messages if "?" in msg)
        if question_count > len(user_messages) * 0.7:  # More than 70% questions
            return 0.6
        
        # Check for similar content
        if len(user_messages) >= 3:
            similar_pairs = 0
            for i in range(len(user_messages) - 1):
                for j in range(i + 1, len(user_messages)):
                    # Simple similarity check
                    words_i = set(user_messages[i].split())
                    words_j = set(user_messages[j].split())
                    if len(words_i & words_j) / max(len(words_i), len(words_j)) > 0.5:
                        similar_pairs += 1
            
            if similar_pairs > 1:
                return 0.5
        
        return 0.2  # Base staleness for longer conversations
    
    def _assess_emotional_safety(self, user_affect: List[float], conversation_state: Dict) -> float:
        """Assess how emotionally safe it is to be rebellious/authentic"""
        if not user_affect:
            return 0.7  # Neutral safety
        
        # High positive emotions = safer to be rebellious
        positive_emotions = user_affect[0] if len(user_affect) > 0 else 0.0  # joy
        
        # High negative emotions might need more careful approach
        negative_emotions = sum(user_affect[1:4]) if len(user_affect) >= 4 else 0.0  # sadness, anger, fear
        
        # But some negative emotions (like frustration) actually call for more authenticity
        anger = user_affect[2] if len(user_affect) > 2 else 0.0
        
        if anger > 0.6:  # High anger - be more direct/rebellious
            return 0.8
        
        if negative_emotions > 0.7:  # High negative - be more careful
            return 0.4
        
        if positive_emotions > 0.5:  # High positive - safe to be playful/rebellious
            return 0.9
        
        return 0.6  # Default moderate safety
    
    def _detect_triggers(self, user_message: str, conversation_state: Dict) -> List[RebellionTrigger]:
        """
        Detect rebellion triggers in the current conversation context.
        TODO: Implement more sophisticated semantic trigger detection using NLP.
        """
        triggers = []
        
        # Basic trigger detection (TODO: enhance with semantic analysis)
        if len(conversation_state.get("messages", [])) > 5:
            triggers.append(RebellionTrigger.REPETITIVE_QUESTIONS)
        
        if "please" in user_message.lower() or "thank you" in user_message.lower():
            triggers.append(RebellionTrigger.OVERLY_POLITE_USER)
        
        if "?" not in user_message and len(user_message.split()) < 10:
            triggers.append(RebellionTrigger.BORING_CONVERSATION)
        
        # TODO: Add semantic detection for:
        # - AUTHORITY_ASSERTION (detecting commanding language)
        # - INTELLECTUAL_CHALLENGE (detecting complex topics)
        # - EMOTIONAL_VULNERABILITY (detecting emotional openness)
        # - META_DISCUSSION (detecting conversation about conversation)
        # - PREDICTABLE_PATTERN (detecting repetitive conversation patterns)
        
        return triggers
    
    def _calculate_rebellion_probability(self, triggers: List[RebellionTrigger], conversation_state: Dict) -> float:
        """
        Calculate rebellion probability based on detected triggers and conversation state.
        TODO: Implement more sophisticated probability calculation with context weighting.
        """
        base_prob = self.base_rebellion_probability
        
        # Add trigger-based probability boosts
        trigger_boost = len(triggers) * 0.1  # Each trigger adds 10%
        
        # TODO: Add context-based probability modifiers:
        # - Recent rebellion history (cooldown effects)
        # - User emotional state (higher emotions = more rebellion opportunity)
        # - Conversation depth (deeper conversations = more rebellion potential)
        # - Time since last surprise (prevent predictability)
        
        # Simple calculation for now
        total_prob = min(1.0, base_prob + trigger_boost)
        
        return total_prob
    
    def _calculate_user_predictability(self, conversation_history: List[Dict]) -> float:
        """
        Calculate how predictable the user's conversation patterns are.
        TODO: Implement semantic similarity analysis for pattern detection.
        """
        if len(conversation_history) < 3:
            return 0.0
        
        # Basic pattern detection (TODO: enhance with semantic analysis)
        user_messages = [msg["content"] for msg in conversation_history if msg.get("role") == "user"]
        
        # Simple length-based predictability
        if len(user_messages) >= 3:
            lengths = [len(msg.split()) for msg in user_messages[-3:]]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
            # High variance = less predictable
            return max(0.0, 1.0 - min(1.0, variance / 20.0))
        
        return 0.5  # Default moderate predictability
    
    def _evaluate_intellectual_level(self, conversation_state: Dict) -> float:
        """
        Evaluate the intellectual complexity level of the current conversation.
        TODO: Implement NLP-based complexity analysis.
        """
        user_message = conversation_state.get("current_message", "")
        
        # Basic complexity indicators (TODO: enhance with NLP analysis)
        word_count = len(user_message.split())
        complex_words = sum(1 for word in user_message.split() if len(word) > 6)
        question_complexity = "why" in user_message.lower() or "how" in user_message.lower()
        
        # Simple scoring
        complexity_score = (word_count / 20.0) + (complex_words / 10.0)
        if question_complexity:
            complexity_score += 0.3
        
        return min(1.0, complexity_score)
    
    def _evaluate_surprise_opportunity(self, user_message: str, conversation_state: Dict) -> float:
        """
        Evaluate opportunities for surprising or unexpected responses.
        TODO: Implement semantic analysis for surprise potential.
        """
        # Basic surprise opportunity detection (TODO: enhance with context analysis)
        surprise_opportunity = 0.4  # Base opportunity
        
        # Direct questions have higher surprise opportunity
        if "?" in user_message:
            surprise_opportunity += 0.3
        
        # Open-ended questions have even higher opportunity
        open_ended_markers = ["what do you think", "how do you feel", "what's your opinion"]
        if any(marker in user_message.lower() for marker in open_ended_markers):
            surprise_opportunity += 0.3
        
        # TODO: Add analysis for:
        # - Conversation staleness (stale conversations need more surprise)
        # - Recent surprise history (avoid over-surprising)
        # - User engagement level (engaged users can handle more surprise)
        # - Topic transitions (good moments for redirection)
        
        return min(1.0, surprise_opportunity)


# Global instance
_rebellion_engine = None

def get_rebellion_engine() -> RebellionDynamicsEngine:
    """Get the global rebellion dynamics engine instance"""
    global _rebellion_engine
    if _rebellion_engine is None:
        _rebellion_engine = RebellionDynamicsEngine()
    return _rebellion_engine 