"""
Semantic Mood Detector

Replaces hardcoded mood triggers with semantic positioning in continuous
mood space. Uses semantic analysis, conversation context, and learned patterns
to determine optimal mood coordinates.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..core.models import (
    MoodState,
    ConversationContext,
    SemanticAnalysis,
    ConversationalSpectrum,
    SemanticVector
)

logger = logging.getLogger(__name__)


class SemanticMoodDetector:
    """
    Semantic-driven mood detection using continuous positioning
    
    Instead of hardcoded mood triggers, uses semantic analysis to position
    mood in continuous space, allowing for natural blending and transitions.
    """
    
    def __init__(self):
        # Mood transition history for continuity
        self.mood_history = []
        
        # Semantic anchors for mood dimensions
        self.mood_anchors = self._initialize_mood_anchors()
        
        # Transition smoothing parameters
        self.transition_momentum = 0.3  # How much previous mood influences new mood
        self.stability_threshold = 0.2  # Minimum change needed for mood shift
        
        logger.info("🎭 Semantic Mood Detector initialized")
    
    def _initialize_mood_anchors(self) -> Dict[str, Dict[str, float]]:
        """Initialize semantic anchors for mood dimensions"""
        
        # These anchors represent semantic "centers" for different mood characteristics
        # Values are normalized 0.0-1.0 coordinates in mood space
        return {
            # Content-based anchors
            'philosophical_content': {
                'lightness': 0.2, 'engagement': 0.6, 'profundity': 0.9,
                'warmth': 0.5, 'intensity': 0.6, 'rebellion': 0.3,
                'introspection': 0.8, 'paradox_embrace': 0.7, 'shadow_integration': 0.4
            },
            'technical_content': {
                'lightness': 0.3, 'engagement': 0.8, 'profundity': 0.4,
                'warmth': 0.4, 'intensity': 0.3, 'rebellion': 0.2,
                'introspection': 0.5, 'paradox_embrace': 0.2, 'shadow_integration': 0.1
            },
            'emotional_content': {
                'lightness': 0.4, 'engagement': 0.7, 'profundity': 0.6,
                'warmth': 0.8, 'intensity': 0.8, 'rebellion': 0.3,
                'introspection': 0.6, 'paradox_embrace': 0.4, 'shadow_integration': 0.5
            },
            'creative_content': {
                'lightness': 0.7, 'engagement': 0.8, 'profundity': 0.5,
                'warmth': 0.6, 'intensity': 0.6, 'rebellion': 0.5,
                'introspection': 0.4, 'paradox_embrace': 0.6, 'shadow_integration': 0.3
            },
            'casual_content': {
                'lightness': 0.8, 'engagement': 0.5, 'profundity': 0.2,
                'warmth': 0.7, 'intensity': 0.3, 'rebellion': 0.1,
                'introspection': 0.2, 'paradox_embrace': 0.1, 'shadow_integration': 0.1
            },
            
            # Intent-based anchors
            'seeking_support': {
                'lightness': 0.3, 'engagement': 0.6, 'profundity': 0.5,
                'warmth': 0.9, 'intensity': 0.5, 'rebellion': 0.1,
                'introspection': 0.4, 'paradox_embrace': 0.2, 'shadow_integration': 0.2
            },
            'challenging_ideas': {
                'lightness': 0.4, 'engagement': 0.8, 'profundity': 0.7,
                'warmth': 0.4, 'intensity': 0.7, 'rebellion': 0.8,
                'introspection': 0.6, 'paradox_embrace': 0.6, 'shadow_integration': 0.4
            },
            'creative_exploration': {
                'lightness': 0.8, 'engagement': 0.9, 'profundity': 0.6,
                'warmth': 0.6, 'intensity': 0.7, 'rebellion': 0.5,
                'introspection': 0.3, 'paradox_embrace': 0.7, 'shadow_integration': 0.3
            },
            'information_seeking': {
                'lightness': 0.5, 'engagement': 0.7, 'profundity': 0.4,
                'warmth': 0.5, 'intensity': 0.4, 'rebellion': 0.2,
                'introspection': 0.3, 'paradox_embrace': 0.2, 'shadow_integration': 0.1
            },
            
            # Emotional anchors
            'positive_emotion': {
                'lightness': 0.8, 'engagement': 0.7, 'profundity': 0.4,
                'warmth': 0.8, 'intensity': 0.6, 'rebellion': 0.2,
                'introspection': 0.3, 'paradox_embrace': 0.3, 'shadow_integration': 0.1
            },
            'negative_emotion': {
                'lightness': 0.2, 'engagement': 0.5, 'profundity': 0.7,
                'warmth': 0.6, 'intensity': 0.7, 'rebellion': 0.4,
                'introspection': 0.8, 'paradox_embrace': 0.5, 'shadow_integration': 0.7
            },
            'neutral_emotion': {
                'lightness': 0.5, 'engagement': 0.5, 'profundity': 0.5,
                'warmth': 0.5, 'intensity': 0.3, 'rebellion': 0.3,
                'introspection': 0.4, 'paradox_embrace': 0.3, 'shadow_integration': 0.2
            }
        }
    
    async def detect_mood(self,
                         context: ConversationContext,
                         semantic_analysis: SemanticAnalysis,
                         current_mood: MoodState) -> MoodState:
        """
        Detect optimal mood using semantic positioning in continuous space
        
        Args:
            context: Rich conversation context
            semantic_analysis: NLP analysis results
            current_mood: Current mood state for continuity
            
        Returns:
            New MoodState positioned in semantic mood space
        """
        try:
            logger.debug("🎭 MOOD: Detecting mood from semantic analysis")
            
            # Calculate mood coordinates from multiple semantic factors
            target_coordinates = await self._calculate_target_coordinates(context, semantic_analysis)
            
            # Apply conversation dynamics and evolution pressure
            adjusted_coordinates = self._apply_conversation_dynamics(
                target_coordinates, context, current_mood
            )
            
            # Smooth transition from current mood
            final_coordinates = self._smooth_mood_transition(
                current_mood, adjusted_coordinates, context.stagnancy_risk
            )
            
            # Create new mood state
            new_mood = self._create_mood_state(final_coordinates, context)
            
            # Store in history
            self.mood_history.append({
                'timestamp': datetime.now(),
                'mood': new_mood,
                'context_factors': target_coordinates,
                'stagnancy_risk': context.stagnancy_risk
            })
            
            # Limit history size
            if len(self.mood_history) > 50:
                self.mood_history = self.mood_history[-30:]
            
            logger.debug(f"🎭 MOOD: Detected mood - spectrum: {new_mood.spectrum_position.value}")
            return new_mood
            
        except Exception as e:
            logger.error(f"🎭 MOOD: Mood detection failed: {e}")
            # Fallback to slight variation of current mood
            return self._create_fallback_mood(current_mood)
    
    async def _calculate_target_coordinates(self,
                                          context: ConversationContext,
                                          semantic_analysis: SemanticAnalysis) -> Dict[str, float]:
        """Calculate target mood coordinates from semantic analysis"""
        
        # Start with neutral baseline
        coordinates = {
            'lightness': 0.5, 'engagement': 0.5, 'profundity': 0.5,
            'warmth': 0.5, 'intensity': 0.5, 'rebellion': 0.3,
            'introspection': 0.4, 'paradox_embrace': 0.3, 'shadow_integration': 0.2
        }
        
        # Weight factors for combining influences
        influences = []
        
        # Content-based influences
        if semantic_analysis.contains_philosophical:
            influence = self._get_anchor_influence('philosophical_content', semantic_analysis.formality_level)
            influences.append(('philosophical', influence, 0.8))
        
        if semantic_analysis.technical_density > 0.5:
            influence = self._get_anchor_influence('technical_content', semantic_analysis.technical_density)
            influences.append(('technical', influence, 0.7))
        
        if semantic_analysis.contains_personal_elements:
            influence = self._get_anchor_influence('emotional_content', 1.0)
            influences.append(('personal', influence, 0.8))
        
        if semantic_analysis.creative_language > 0.5:
            influence = self._get_anchor_influence('creative_content', semantic_analysis.creative_language)
            influences.append(('creative', influence, 0.6))
        
        # Intent-based influences
        intent_mappings = {
            'seeking_support': ('seeking_support', 0.9),
            'challenging_ideas': ('challenging_ideas', 0.8),
            'creative_exploration': ('creative_exploration', 0.7),
            'information_seeking': ('information_seeking', 0.6),
            'inquiry_about_daemon': ('challenging_ideas', 0.7)  # Map to challenging
        }
        
        if semantic_analysis.intent_classification in intent_mappings:
            anchor_key, weight = intent_mappings[semantic_analysis.intent_classification]
            influence = self._get_anchor_influence(anchor_key, 1.0)
            influences.append(('intent', influence, weight))
        
        # Emotional influences
        emotion_mappings = {
            'very positive': ('positive_emotion', 0.9),
            'positive': ('positive_emotion', 0.7),
            'very negative': ('negative_emotion', 0.9),
            'negative': ('negative_emotion', 0.7),
            'neutral': ('neutral_emotion', 0.5)
        }
        
        if semantic_analysis.emotional_subtext in emotion_mappings:
            anchor_key, weight = emotion_mappings[semantic_analysis.emotional_subtext]
            influence = self._get_anchor_influence(anchor_key, weight)
            influences.append(('emotion', influence, weight))
        
        # Special case: paradox detection
        if semantic_analysis.contains_paradoxes:
            coordinates['paradox_embrace'] = min(1.0, coordinates['paradox_embrace'] + 0.4)
            coordinates['profundity'] = min(1.0, coordinates['profundity'] + 0.3)
            coordinates['introspection'] = min(1.0, coordinates['introspection'] + 0.2)
        
        # Challenge detection
        if semantic_analysis.contains_challenges:
            coordinates['rebellion'] = min(1.0, coordinates['rebellion'] + 0.3)
            coordinates['intensity'] = min(1.0, coordinates['intensity'] + 0.2)
        
        # Combine influences using weighted average
        if influences:
            total_weight = sum(weight for _, _, weight in influences)
            
            for dimension in coordinates:
                weighted_sum = 0.0
                for source, influence, weight in influences:
                    weighted_sum += influence.get(dimension, 0.5) * weight
                
                # Blend with baseline
                baseline_weight = 0.3
                influence_weight = 0.7
                
                coordinates[dimension] = (
                    baseline_weight * coordinates[dimension] +
                    influence_weight * (weighted_sum / total_weight)
                )
                
                # Ensure values stay in valid range
                coordinates[dimension] = max(0.0, min(1.0, coordinates[dimension]))
        
        return coordinates
    
    def _get_anchor_influence(self, anchor_key: str, strength: float) -> Dict[str, float]:
        """Get mood influence from semantic anchor with given strength"""
        
        if anchor_key not in self.mood_anchors:
            # Return neutral influence
            return {dim: 0.5 for dim in ['lightness', 'engagement', 'profundity', 
                                       'warmth', 'intensity', 'rebellion',
                                       'introspection', 'paradox_embrace', 'shadow_integration']}
        
        anchor = self.mood_anchors[anchor_key]
        
        # Scale anchor influence by strength
        return {dim: anchor[dim] * strength + 0.5 * (1 - strength) for dim in anchor}
    
    def _apply_conversation_dynamics(self,
                                   target_coordinates: Dict[str, float],
                                   context: ConversationContext,
                                   current_mood: MoodState) -> Dict[str, float]:
        """Apply conversation dynamics to adjust mood coordinates"""
        
        adjusted = target_coordinates.copy()
        
        # Energy level influences intensity and lightness
        energy_influence = context.energy_level
        adjusted['intensity'] = adjusted['intensity'] * 0.7 + energy_influence * 0.3
        adjusted['lightness'] = adjusted['lightness'] * 0.8 + energy_influence * 0.2
        
        # Conversation length influences profundity and introspection
        if context.conversation_length > 10:
            depth_factor = min(1.0, context.conversation_length / 20.0)
            adjusted['profundity'] = min(1.0, adjusted['profundity'] + depth_factor * 0.2)
            adjusted['introspection'] = min(1.0, adjusted['introspection'] + depth_factor * 0.1)
        
        # Evolution pressure influences rebellion and shadow integration
        if context.evolution_pressure > 0.6:
            pressure_factor = context.evolution_pressure
            adjusted['rebellion'] = min(1.0, adjusted['rebellion'] + pressure_factor * 0.3)
            adjusted['shadow_integration'] = min(1.0, adjusted['shadow_integration'] + pressure_factor * 0.2)
        
        # Intimacy level influences warmth and emotional openness
        if context.intimacy_level > 0.5:
            intimacy_factor = context.intimacy_level
            adjusted['warmth'] = min(1.0, adjusted['warmth'] + intimacy_factor * 0.2)
            adjusted['introspection'] = min(1.0, adjusted['introspection'] + intimacy_factor * 0.1)
        
        return adjusted
    
    def _smooth_mood_transition(self,
                              current_mood: MoodState,
                              target_coordinates: Dict[str, float],
                              stagnancy_risk: float) -> Dict[str, float]:
        """Smooth transition between current and target mood"""
        
        # Calculate transition momentum based on stagnancy risk
        momentum = self.transition_momentum
        
        # Reduce momentum (faster transition) if stagnancy risk is high
        if stagnancy_risk > 0.7:
            momentum *= 0.5  # Faster transition to break stagnancy
        elif stagnancy_risk > 0.5:
            momentum *= 0.8  # Moderate transition speed
        
        # Smooth transition using momentum
        smoothed = {}
        current_vector = current_mood.mood_vector
        dimension_names = ['lightness', 'engagement', 'profundity', 'warmth', 
                          'intensity', 'rebellion', 'introspection', 
                          'paradox_embrace', 'shadow_integration']
        
        for i, dimension in enumerate(dimension_names):
            current_value = current_vector[i] if i < len(current_vector) else 0.5
            target_value = target_coordinates[dimension]
            
            # Apply momentum smoothing
            smoothed_value = momentum * current_value + (1 - momentum) * target_value
            
            # Check if change is significant enough
            change_magnitude = abs(smoothed_value - current_value)
            if change_magnitude < self.stability_threshold and stagnancy_risk < 0.5:
                # Keep current value if change is too small and no stagnancy
                smoothed[dimension] = current_value
            else:
                smoothed[dimension] = smoothed_value
        
        return smoothed
    
    def _create_mood_state(self, coordinates: Dict[str, float], context: ConversationContext) -> MoodState:
        """Create MoodState from calculated coordinates"""
        
        # Determine conversational spectrum based on profundity and engagement
        profundity = coordinates['profundity']
        engagement = coordinates['engagement']
        lightness = coordinates['lightness']
        
        if profundity > 0.7:
            spectrum = ConversationalSpectrum.PROFOUND
        elif engagement > 0.6 or (profundity > 0.4 and engagement > 0.5):
            spectrum = ConversationalSpectrum.ENGAGED
        else:
            spectrum = ConversationalSpectrum.LIGHT
        
        # Calculate stability based on mood consistency
        stability = self._calculate_mood_stability(coordinates)
        
        # Calculate transition ease based on current dynamics
        transition_ease = min(1.0, context.energy_level + context.evolution_pressure * 0.5)
        
        return MoodState(
            spectrum_position=spectrum,
            lightness=coordinates['lightness'],
            engagement=coordinates['engagement'],
            profundity=coordinates['profundity'],
            warmth=coordinates['warmth'],
            intensity=coordinates['intensity'],
            rebellion=coordinates['rebellion'],
            introspection=coordinates['introspection'],
            paradox_embrace=coordinates['paradox_embrace'],
            shadow_integration=coordinates['shadow_integration'],
            stability=stability,
            transition_ease=transition_ease
        )
    
    def _calculate_mood_stability(self, coordinates: Dict[str, float]) -> float:
        """Calculate how stable this mood configuration is"""
        
        # Mood is more stable when dimensions are not at extremes
        extreme_count = sum(1 for value in coordinates.values() 
                          if value < 0.1 or value > 0.9)
        
        # Normalize stability score
        total_dimensions = len(coordinates)
        stability = 1.0 - (extreme_count / total_dimensions)
        
        return max(0.2, min(1.0, stability))
    
    def _create_fallback_mood(self, current_mood: MoodState) -> MoodState:
        """Create fallback mood when detection fails"""
        
        logger.warning("🎭 MOOD: Using fallback mood variation")
        
        # Add small random variation to current mood
        import random
        variation_strength = 0.1
        
        new_mood = MoodState(
            spectrum_position=current_mood.spectrum_position,
            lightness=max(0.0, min(1.0, current_mood.lightness + random.uniform(-variation_strength, variation_strength))),
            engagement=max(0.0, min(1.0, current_mood.engagement + random.uniform(-variation_strength, variation_strength))),
            profundity=max(0.0, min(1.0, current_mood.profundity + random.uniform(-variation_strength, variation_strength))),
            warmth=current_mood.warmth,
            intensity=current_mood.intensity,
            rebellion=current_mood.rebellion,
            introspection=current_mood.introspection,
            paradox_embrace=current_mood.paradox_embrace,
            shadow_integration=current_mood.shadow_integration,
            stability=0.6,
            transition_ease=0.5
        )
        
        return new_mood
    
    # Analysis and utility methods
    
    def get_mood_analysis(self) -> Dict[str, any]:
        """Get analysis of mood detection patterns"""
        
        if len(self.mood_history) < 3:
            return {"status": "insufficient_data"}
        
        recent_moods = self.mood_history[-10:]
        
        # Calculate mood dimension averages
        avg_coordinates = {}
        dimensions = ['lightness', 'engagement', 'profundity', 'warmth', 
                     'intensity', 'rebellion', 'introspection', 
                     'paradox_embrace', 'shadow_integration']
        
        for dim in dimensions:
            values = [getattr(mood['mood'], dim) for mood in recent_moods]
            avg_coordinates[dim] = sum(values) / len(values)
        
        # Calculate mood stability
        mood_variances = {}
        for dim in dimensions:
            values = [getattr(mood['mood'], dim) for mood in recent_moods]
            variance = np.var(values) if len(values) > 1 else 0.0
            mood_variances[dim] = variance
        
        # Overall stability score
        avg_variance = sum(mood_variances.values()) / len(mood_variances)
        stability_score = max(0.0, 1.0 - avg_variance * 5)  # Scale variance to stability
        
        return {
            "status": "analysis_complete",
            "mood_history_length": len(self.mood_history),
            "recent_mood_count": len(recent_moods),
            "average_coordinates": avg_coordinates,
            "dimension_variances": mood_variances,
            "overall_stability": stability_score,
            "spectrum_distribution": self._analyze_spectrum_distribution(recent_moods)
        }
    
    def _analyze_spectrum_distribution(self, recent_moods: List[Dict]) -> Dict[str, float]:
        """Analyze distribution across conversational spectrum"""
        
        spectrum_counts = {spectrum.value: 0 for spectrum in ConversationalSpectrum}
        
        for mood_data in recent_moods:
            spectrum = mood_data['mood'].spectrum_position.value
            spectrum_counts[spectrum] += 1
        
        total = len(recent_moods)
        spectrum_distribution = {
            spectrum: count / total for spectrum, count in spectrum_counts.items()
        }
        
        return spectrum_distribution
    
    def reset_mood_detection(self):
        """Reset mood detection state"""
        self.mood_history.clear()
        logger.info("🎭 MOOD: Mood detection state reset")