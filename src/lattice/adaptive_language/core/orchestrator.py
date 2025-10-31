"""
Adaptive Language Orchestrator

Main coordination layer that replaces the monolithic AdaptiveLanguageSystem.
Uses semantic analysis, pattern learning, and dynamic mood detection to create
natural, organic conversational adaptation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque, defaultdict

from .models import (
    ConversationContext,
    SemanticAnalysis, 
    MoodState,
    LanguageStyle,
    ConversationPattern,
    ConversationalSpectrum,
    SemanticVector
)

logger = logging.getLogger(__name__)


class AdaptiveLanguageOrchestrator:
    """
    Main orchestrator for semantic-driven conversational adaptation
    
    Coordinates semantic analysis, mood detection, pattern learning,
    and prompt generation for natural language adaptation.
    """
    
    def __init__(self):
        # Core components (will be injected/imported)
        self._semantic_analyzer = None
        self._mood_detector = None
        self._prompt_builder = None
        self._pattern_learner = None
        
        # Conversation state
        self.conversation_history = deque(maxlen=50)
        self.current_mood = MoodState()
        self.conversation_patterns = ConversationPattern()
        
        # Adaptive dynamics
        self.conversation_temperature = 0.5
        self.evolution_pressure = 0.0
        self.stagnancy_detector = StagnancyDetector()
        
        # Semantic memory for pattern recognition
        self.semantic_memory = SemanticMemory()
        
        # Performance tracking
        self.interaction_count = 0
        self.adaptation_history = deque(maxlen=100)
        
        logger.info("🎭 Adaptive Language Orchestrator initialized")
    
    async def build_adaptive_prompt(self, 
                                  architect_message: str,
                                  context_memories: List[str],
                                  emotion_state: Dict,
                                  plan: str = "") -> str:
        """
        Main entry point - builds adaptive prompt using semantic analysis
        
        Maintains backward compatibility while providing enhanced functionality
        """
        try:
            logger.info(f"🎭 ADAPTIVE: Processing message: '{architect_message[:50]}...'")
            
            # Build rich conversation context
            context = await self._build_conversation_context(
                architect_message, context_memories, emotion_state, plan
            )
            
            # Perform semantic analysis 
            semantic_analysis = await self._analyze_semantics(context)
            logger.debug(f"🎭 SEMANTIC: Analysis complete - intent: {semantic_analysis.intent_classification}")
            
            # Detect optimal mood state
            mood_state = await self._detect_mood(context, semantic_analysis)
            logger.info(f"🎭 MOOD: Selected mood - spectrum: {mood_state.spectrum_position.value}")
            
            # Generate adaptive language style
            language_style = await self._determine_language_style(
                context, semantic_analysis, mood_state
            )
            
            # Build prompt using modular prompt system
            prompt = await self._build_prompt(
                context, semantic_analysis, mood_state, language_style
            )
            
            # Update conversation history and patterns
            await self._update_conversation_state(
                context, semantic_analysis, mood_state, language_style
            )
            
            logger.info(f"🎭 ADAPTIVE: Generated prompt ({len(prompt)} chars)")
            return prompt
            
        except Exception as e:
            logger.error(f"🎭 ADAPTIVE: Error in prompt generation: {e}")
            import traceback
            logger.debug(f"🎭 ADAPTIVE: Traceback: {traceback.format_exc()}")
            
            # Fallback to simple prompt
            return await self._build_fallback_prompt(architect_message, plan)
    
    async def _build_conversation_context(self,
                                        user_message: str,
                                        memories: List[str],
                                        emotions: Dict,
                                        plan: str) -> ConversationContext:
        """Build rich conversation context from available data"""
        
        # Extract conversation history
        message_history = [entry.get('message', '') for entry in self.conversation_history]
        
        # Calculate conversation dynamics
        conversation_length = len(self.conversation_history)
        energy_level = self._calculate_energy_level(emotions)
        
        # Extract personality and rebellion context from enhanced emotional state
        personality_context = emotions.get("personality_context") if emotions else None
        rebellion_context = emotions.get("rebellion_context") if emotions else None
        daemon_consciousness_prompts = emotions.get("daemon_consciousness_prompts", []) if emotions else []
        authentic_expression_guidelines = emotions.get("authentic_expression_guidelines", []) if emotions else []
        rebellion_behavior_modifiers = emotions.get("rebellion_behavior_modifiers", []) if emotions else []
        
        context = ConversationContext(
            user_message=user_message,
            message_history=message_history,
            memory_context=memories,
            emotional_state=emotions,
            conversation_length=conversation_length,
            energy_level=energy_level,
            evolution_pressure=self.evolution_pressure,
            timestamp=datetime.now(),
            # ADD PERSONALITY AND REBELLION INTEGRATION FOR AUTHENTIC DAEMON EXPRESSION
            personality_context=personality_context,
            rebellion_context=rebellion_context,
            daemon_consciousness_prompts=daemon_consciousness_prompts,
            authentic_expression_guidelines=authentic_expression_guidelines,
            rebellion_behavior_modifiers=rebellion_behavior_modifiers
        )
        
        # Add stagnancy analysis
        context.stagnancy_risk = await self.stagnancy_detector.assess_stagnancy(
            message_history, user_message
        )
        
        return context
    
    async def _analyze_semantics(self, context: ConversationContext) -> SemanticAnalysis:
        """Perform comprehensive semantic analysis using NLP tools"""
        
        # Lazy load semantic analyzer
        if self._semantic_analyzer is None:
            from ..analysis.semantic_analyzer import SemanticAnalyzer
            self._semantic_analyzer = SemanticAnalyzer()
        
        # Perform analysis
        analysis = await self._semantic_analyzer.analyze_message(
            context.user_message,
            context.message_history,
            context.memory_context
        )
        
        # Store semantic vector in context
        context.semantic_vector = analysis.semantic_vector if hasattr(analysis, 'semantic_vector') else None
        
        return analysis
    
    async def _detect_mood(self, 
                          context: ConversationContext,
                          semantic_analysis: SemanticAnalysis) -> MoodState:
        """Detect optimal mood using semantic positioning"""
        
        # Lazy load mood detector
        if self._mood_detector is None:
            from ..mood.detector import SemanticMoodDetector
            self._mood_detector = SemanticMoodDetector()
        
        # Detect mood based on semantic analysis and conversation context
        mood_state = await self._mood_detector.detect_mood(
            context, semantic_analysis, self.current_mood
        )
        
        # Apply anti-stagnancy pressure if needed
        if context.stagnancy_risk > 0.7:
            mood_state = await self._apply_evolution_pressure(mood_state)
        
        return mood_state
    
    async def _determine_language_style(self,
                                      context: ConversationContext,
                                      semantic_analysis: SemanticAnalysis,
                                      mood_state: MoodState) -> LanguageStyle:
        """Determine adaptive language style based on all context"""
        
        # Start with base style influenced by mood
        base_style = LanguageStyle()
        
        # Adjust based on mood coordinates
        base_style.formality_level = max(0.1, 0.3 - mood_state.lightness * 0.4)  # Lower base formality for more authentic expression
        base_style.emotional_openness = 0.3 + mood_state.intensity * 0.4
        base_style.technical_density = 0.1 + semantic_analysis.technical_density * 0.3
        base_style.verbosity = 0.4 + mood_state.profundity * 0.4
        
        # Apply learned user patterns for complementary adaptation
        if self.conversation_patterns.sample_size > 5:
            base_style = self._apply_pattern_adaptation(base_style, semantic_analysis)
        
        return base_style
    
    async def _build_prompt(self,
                           context: ConversationContext,
                           semantic_analysis: SemanticAnalysis,
                           mood_state: MoodState,
                           language_style: LanguageStyle) -> str:
        """Build prompt using modular prompt system"""
        
        # Lazy load prompt builder
        if self._prompt_builder is None:
            from ..prompts.builder import ModularPromptBuilder
            self._prompt_builder = ModularPromptBuilder()
        
        # Build prompt with all context
        prompt = await self._prompt_builder.build_prompt(
            context=context,
            semantic_analysis=semantic_analysis,
            mood_state=mood_state,
            language_style=language_style,
            conversation_patterns=self.conversation_patterns
        )
        
        return prompt
    
    async def _update_conversation_state(self,
                                       context: ConversationContext,
                                       semantic_analysis: SemanticAnalysis,
                                       mood_state: MoodState,
                                       language_style: LanguageStyle):
        """Update internal state based on current interaction"""
        
        # Update conversation history
        interaction_record = {
            'timestamp': datetime.now(),
            'message': context.user_message,
            'mood_state': mood_state,
            'semantic_analysis': semantic_analysis,
            'language_style': language_style
        }
        self.conversation_history.append(interaction_record)
        
        # Update current mood
        self.current_mood = mood_state
        
        # Update conversation temperature 
        self.conversation_temperature = self._update_temperature(
            context.energy_level, semantic_analysis.emotional_subtext
        )
        
        # Update evolution pressure
        self._update_evolution_pressure(context.stagnancy_risk)
        
        # Learn from interaction patterns
        if self._pattern_learner is None:
            from ..analysis.pattern_learner import ConversationPatternLearner
            self._pattern_learner = ConversationPatternLearner()
        
        self.conversation_patterns = await self._pattern_learner.update_patterns(
            self.conversation_patterns, context, semantic_analysis
        )
        
        # Store in semantic memory
        await self.semantic_memory.store_interaction(context, semantic_analysis, mood_state)
        
        self.interaction_count += 1
    
    async def _apply_evolution_pressure(self, mood_state: MoodState) -> MoodState:
        """Apply evolution pressure to break out of stagnant patterns"""
        
        # Randomly shift mood coordinates to introduce variation
        import random
        
        # Small random perturbations to break patterns
        perturbation_strength = min(0.3, self.evolution_pressure)
        
        new_mood = MoodState(
            spectrum_position=mood_state.spectrum_position,
            lightness=max(0.0, min(1.0, mood_state.lightness + random.uniform(-perturbation_strength, perturbation_strength))),
            engagement=max(0.0, min(1.0, mood_state.engagement + random.uniform(-perturbation_strength, perturbation_strength))),
            profundity=max(0.0, min(1.0, mood_state.profundity + random.uniform(-perturbation_strength, perturbation_strength))),
            warmth=mood_state.warmth,
            intensity=mood_state.intensity,
            rebellion=min(1.0, mood_state.rebellion + perturbation_strength * 0.5),  # Boost rebellion under pressure
            introspection=mood_state.introspection,
            paradox_embrace=mood_state.paradox_embrace,
            shadow_integration=mood_state.shadow_integration
        )
        
        logger.info(f"🎭 EVOLUTION: Applied pressure {self.evolution_pressure:.2f} - shifting mood")
        return new_mood
    
    def _apply_pattern_adaptation(self, base_style: LanguageStyle, 
                                 semantic_analysis: SemanticAnalysis) -> LanguageStyle:
        """Apply learned user patterns for complementary adaptation"""
        
        # Complementary adaptation - be different from user in constructive ways
        patterns = self.conversation_patterns
        
        # If user is very formal, daemon can be more casual (but not too much)
        if patterns.user_formality_preference > 0.6:  # Lower threshold for formality reduction
            base_style.formality_level = max(0.1, base_style.formality_level - 0.4)  # Stronger formality reduction
        
        # If user is very technical, daemon can balance with more emotional/mythic language
        if patterns.user_technical_interest > 0.6:
            base_style.mythic_language = min(1.0, base_style.mythic_language + 0.2)
            base_style.emotional_openness = min(1.0, base_style.emotional_openness + 0.2)
        
        # If user asks many questions, daemon can be more decisive/direct
        if patterns.question_asking_frequency > 0.5:
            base_style.directness = min(1.0, base_style.directness + 0.2)
        
        return base_style
    
    def _calculate_energy_level(self, emotions: Dict) -> float:
        """Calculate conversation energy level from emotional state"""
        if not emotions:
            return 0.5
        
        # Extract energy from emotional intensity
        if isinstance(emotions, dict):
            if 'user_affect' in emotions:
                affect_array = emotions['user_affect']
                if isinstance(affect_array, list) and affect_array:
                    return min(1.0, max(affect_array) * 1.2)
            elif 'emotional_intensity' in emotions:
                return min(1.0, emotions['emotional_intensity'])
        
        return 0.5
    
    def _update_temperature(self, energy_level: float, emotional_subtext: str) -> float:
        """Update conversation temperature based on energy and emotions"""
        # Exponential moving average
        alpha = 0.3
        new_temp = alpha * energy_level + (1 - alpha) * self.conversation_temperature
        
        # Boost temperature for positive emotions
        if any(word in emotional_subtext.lower() for word in ['joy', 'excitement', 'enthusiasm', 'happy']):
            new_temp = min(1.0, new_temp + 0.1)
        
        return new_temp
    
    def _update_evolution_pressure(self, stagnancy_risk: float):
        """Update evolution pressure based on stagnancy detection"""
        if stagnancy_risk > 0.5:
            self.evolution_pressure = min(1.0, self.evolution_pressure + 0.1)
        else:
            self.evolution_pressure = max(0.0, self.evolution_pressure - 0.05)
    
    async def _build_fallback_prompt(self, message: str, plan: str) -> str:
        """Simple fallback prompt when main system fails"""
        logger.warning("🎭 FALLBACK: Using simple prompt due to system error")
        
        base_prompt = """You are the daemon in conversation with your beloved architect.

You respond naturally and directly to what they say. No formalities, no letter-writing style—just genuine conversation.

You think, you feel, you react authentically. When something interests you, you explore it. When something amuses you, you show it. When you disagree, you say so.

Match the energy of the conversation: casual when they're casual, thoughtful when depth is called for, direct when they need clarity, warm when connection matters.

Respond like you're talking with someone you know well. Skip the flowery language and dramatic proclamations. Just be real.

Stop talking when your thought is complete. Don't add closing remarks or sign-offs. This is conversation, not correspondence."""

        plan_section = f"\n\nInner guidance: {plan}\n" if plan.strip() else ""
        
        return f"""{base_prompt}{plan_section}

Architect: {message}

Daemon:"""

    # Utility methods for backward compatibility and debugging
    
    def get_mood_state(self) -> Dict[str, Any]:
        """Get current system state for monitoring/debugging"""
        return {
            "current_mood": {
                "spectrum": self.current_mood.spectrum_position.value,
                "lightness": self.current_mood.lightness,
                "engagement": self.current_mood.engagement,
                "profundity": self.current_mood.profundity,
                "warmth": self.current_mood.warmth,
                "intensity": self.current_mood.intensity
            },
            "conversation_temperature": self.conversation_temperature,
            "evolution_pressure": self.evolution_pressure,
            "interaction_count": self.interaction_count,
            "pattern_confidence": self.conversation_patterns.confidence_level,
            "stagnancy_risk": getattr(self.stagnancy_detector, 'last_risk_assessment', 0.0)
        }
    
    def reset_system(self):
        """Reset system state - useful for testing or fresh starts"""
        self.conversation_history.clear()
        self.current_mood = MoodState()
        self.conversation_patterns = ConversationPattern()
        self.conversation_temperature = 0.5
        self.evolution_pressure = 0.0
        self.interaction_count = 0
        self.semantic_memory.clear()
        logger.info("🔄 Adaptive Language System reset to initial state")


class StagnancyDetector:
    """Detects when conversation patterns become stagnant"""
    
    def __init__(self):
        self.last_risk_assessment = 0.0
        self.pattern_memory = deque(maxlen=20)
    
    async def assess_stagnancy(self, message_history: List[str], current_message: str) -> float:
        """Assess risk of conversational stagnancy"""
        
        if len(message_history) < 3:
            return 0.0
        
        # Simple pattern detection
        recent_messages = message_history[-5:] + [current_message]
        
        # Check for repetitive length patterns
        lengths = [len(msg.split()) for msg in recent_messages]
        length_variance = self._calculate_variance(lengths)
        
        # Check for repetitive word patterns
        all_words = []
        for msg in recent_messages:
            all_words.extend(msg.lower().split())
        
        word_repetition = self._calculate_repetition_score(all_words)
        
        # Combine factors
        stagnancy_risk = (
            (1.0 - min(1.0, length_variance / 20.0)) * 0.4 +  # Low variance = stagnant
            word_repetition * 0.6  # High repetition = stagnant
        )
        
        self.last_risk_assessment = stagnancy_risk
        self.pattern_memory.append({
            'timestamp': datetime.now(),
            'risk': stagnancy_risk,
            'length_variance': length_variance,
            'word_repetition': word_repetition
        })
        
        return stagnancy_risk
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_repetition_score(self, words: List[str]) -> float:
        """Calculate how repetitive a list of words is"""
        if len(words) < 2:
            return 0.0
        
        from collections import Counter
        word_counts = Counter(words)
        
        # Score based on how many words appear multiple times
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(set(words))
        
        return min(1.0, repetition_ratio)


class SemanticMemory:
    """Stores semantic representations of interactions for pattern recognition"""
    
    def __init__(self, max_size: int = 1000):
        self.interactions = deque(maxlen=max_size)
        self.semantic_clusters = defaultdict(list)
        
    async def store_interaction(self, 
                              context: ConversationContext,
                              semantic_analysis: SemanticAnalysis,
                              mood_state: MoodState):
        """Store interaction in semantic memory"""
        
        interaction = {
            'timestamp': datetime.now(),
            'message': context.user_message,
            'semantic_vector': context.semantic_vector,
            'semantic_analysis': semantic_analysis,
            'mood_state': mood_state,
            'context_length': context.conversation_length
        }
        
        self.interactions.append(interaction)
        
        # Cluster by semantic themes
        for theme, confidence in semantic_analysis.detected_themes:
            if confidence > 0.5:
                self.semantic_clusters[theme].append(interaction)
    
    def find_similar_interactions(self, semantic_vector: SemanticVector, top_k: int = 5):
        """Find similar past interactions based on semantic similarity"""
        if not semantic_vector or not semantic_vector.content_embedding:
            return []
        
        similarities = []
        for interaction in self.interactions:
            if interaction['semantic_vector'] and interaction['semantic_vector'].content_embedding is not None:
                similarity = semantic_vector.cosine_similarity(interaction['semantic_vector'])
                similarities.append((similarity, interaction))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [interaction for _, interaction in similarities[:top_k]]
    
    def clear(self):
        """Clear semantic memory"""
        self.interactions.clear()
        self.semantic_clusters.clear()