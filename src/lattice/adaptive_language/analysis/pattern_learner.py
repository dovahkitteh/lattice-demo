"""
Conversation Pattern Learner

Uses scikit-learn and statistical analysis to learn user communication patterns
and develop complementary conversational styles for the daemon.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    logging.warning("scikit-learn not available for pattern learning")
    TfidfVectorizer = None
    KMeans = None

from ..core.models import (
    ConversationPattern,
    ConversationContext, 
    SemanticAnalysis,
    LanguageStyle
)

logger = logging.getLogger(__name__)


class ConversationPatternLearner:
    """
    Learns user communication patterns through statistical analysis
    and develops complementary daemon conversation styles.
    """
    
    def __init__(self):
        # Pattern learning components
        self._tfidf_vectorizer = None
        self._style_clusters = None
        
        # User pattern tracking
        self.user_messages = deque(maxlen=200)  # Keep recent messages for analysis
        self.interaction_features = deque(maxlen=200)
        
        # Learning history
        self.learning_sessions = []
        self.pattern_evolution = defaultdict(list)
        
        # Complementary style development
        self.daemon_response_history = deque(maxlen=200)
        self.effectiveness_scores = deque(maxlen=200)
        
        logger.info("🧠 Pattern Learner initialized")
    
    async def update_patterns(self, 
                            current_patterns: ConversationPattern,
                            context: ConversationContext,
                            semantic_analysis: SemanticAnalysis) -> ConversationPattern:
        """
        Update conversation patterns based on new interaction data
        
        Args:
            current_patterns: Current pattern understanding
            context: Current conversation context
            semantic_analysis: Analysis of current message
            
        Returns:
            Updated ConversationPattern with new insights
        """
        try:
            logger.debug("🧠 PATTERN: Updating conversation patterns")
            
            # Store new interaction data
            await self._store_interaction_data(context, semantic_analysis)
            
            # Update user communication patterns
            updated_patterns = await self._update_user_patterns(current_patterns, context, semantic_analysis)
            
            # Develop complementary daemon style
            updated_patterns.complementary_style = await self._develop_complementary_style(
                updated_patterns, context, semantic_analysis
            )
            
            # Update pattern confidence
            updated_patterns.confidence_level = self._calculate_pattern_confidence(updated_patterns)
            updated_patterns.last_updated = datetime.now()
            
            # Track pattern evolution
            await self._track_pattern_evolution(updated_patterns)
            
            logger.debug(f"🧠 PATTERN: Updated patterns - confidence: {updated_patterns.confidence_level:.2f}")
            return updated_patterns
            
        except Exception as e:
            logger.error(f"🧠 PATTERN: Pattern update failed: {e}")
            # Return original patterns on error
            return current_patterns
    
    async def _store_interaction_data(self, 
                                    context: ConversationContext,
                                    semantic_analysis: SemanticAnalysis):
        """Store interaction data for pattern analysis"""
        
        # Store user message
        self.user_messages.append({
            'timestamp': datetime.now(),
            'message': context.user_message,
            'length': len(context.user_message.split()),
            'semantic_analysis': semantic_analysis,
            'context_length': context.conversation_length
        })
        
        # Extract interaction features
        features = self._extract_interaction_features(context, semantic_analysis)
        self.interaction_features.append(features)
    
    def _extract_interaction_features(self, 
                                    context: ConversationContext,
                                    semantic_analysis: SemanticAnalysis) -> Dict[str, float]:
        """Extract numerical features from interaction for ML analysis"""
        
        features = {
            # Message characteristics
            'message_length': len(context.user_message.split()),
            'formality_level': semantic_analysis.formality_level,
            'technical_density': semantic_analysis.technical_density,
            'creative_language': semantic_analysis.creative_language,
            'syntactic_complexity': semantic_analysis.syntactic_complexity,
            'lexical_diversity': semantic_analysis.lexical_diversity,
            
            # Content characteristics
            'contains_questions': float(semantic_analysis.contains_questions),
            'contains_challenges': float(semantic_analysis.contains_challenges),
            'contains_personal': float(semantic_analysis.contains_personal_elements),
            'contains_philosophical': float(semantic_analysis.contains_philosophical),
            
            # Thematic strength
            'theme_count': len(semantic_analysis.detected_themes),
            'max_theme_confidence': max([conf for _, conf in semantic_analysis.detected_themes], default=0.0),
            
            # Conversational dynamics
            'conversation_length': context.conversation_length,
            'energy_level': context.energy_level,
            'intimacy_level': context.intimacy_level,
            
            # Temporal features
            'hour_of_day': datetime.now().hour / 24.0,
            'day_of_week': datetime.now().weekday() / 7.0
        }
        
        return features
    
    async def _update_user_patterns(self, 
                                   current_patterns: ConversationPattern,
                                   context: ConversationContext,
                                   semantic_analysis: SemanticAnalysis) -> ConversationPattern:
        """Update understanding of user communication patterns"""
        
        # Use exponential moving average for gradual adaptation
        alpha = 0.1  # Learning rate - adjust for faster/slower adaptation
        
        # Update complexity preferences
        message_complexity = len(context.user_message.split()) / 20.0  # Normalize
        current_patterns.user_preferred_complexity = (
            (1 - alpha) * current_patterns.user_preferred_complexity + 
            alpha * min(1.0, message_complexity)
        )
        
        # Update formality preference
        current_patterns.user_formality_preference = (
            (1 - alpha) * current_patterns.user_formality_preference +
            alpha * semantic_analysis.formality_level
        )
        
        # Update emotional expression level
        emotional_indicators = (
            float(semantic_analysis.contains_personal_elements) * 0.4 +
            (1.0 if semantic_analysis.emotional_subtext not in ['neutral'] else 0.0) * 0.6
        )
        current_patterns.user_emotional_expression = (
            (1 - alpha) * current_patterns.user_emotional_expression +
            alpha * emotional_indicators
        )
        
        # Update technical interest
        current_patterns.user_technical_interest = (
            (1 - alpha) * current_patterns.user_technical_interest +
            alpha * semantic_analysis.technical_density
        )
        
        # Update philosophical engagement
        philosophical_score = (
            float(semantic_analysis.contains_philosophical) * 0.6 +
            (len(semantic_analysis.detected_themes) / 5.0) * 0.4  # Theme richness
        )
        current_patterns.user_philosophical_engagement = (
            (1 - alpha) * current_patterns.user_philosophical_engagement +
            alpha * min(1.0, philosophical_score)
        )
        
        # Update conversation style metrics
        await self._update_conversation_metrics(current_patterns, context, semantic_analysis)
        
        # Update preferred themes using clustering if we have enough data
        await self._update_theme_preferences(current_patterns, semantic_analysis)
        
        return current_patterns
    
    async def _update_conversation_metrics(self,
                                         patterns: ConversationPattern,
                                         context: ConversationContext,
                                         semantic_analysis: SemanticAnalysis):
        """Update conversation-level metrics"""
        
        alpha = 0.15  # Slightly higher learning rate for behavioral patterns
        
        # Update typical conversation length
        if context.conversation_length > 0:
            patterns.typical_conversation_length = int(
                (1 - alpha) * patterns.typical_conversation_length +
                alpha * context.conversation_length
            )
        
        # Update preferred response length (estimate based on message length)
        message_length = len(context.user_message.split())
        patterns.preferred_response_length = int(
            (1 - alpha) * patterns.preferred_response_length +
            alpha * min(300, message_length * 1.5)  # Daemon can be more verbose
        )
        
        # Update question asking frequency
        question_score = float(semantic_analysis.contains_questions)
        patterns.question_asking_frequency = (
            (1 - alpha) * patterns.question_asking_frequency +
            alpha * question_score
        )
        
        # Update conversation rhythm based on response patterns
        if len(self.interaction_features) >= 3:
            recent_features = list(self.interaction_features)[-3:]
            avg_complexity = np.mean([f['syntactic_complexity'] for f in recent_features])
            
            if avg_complexity > 0.7:
                patterns.conversation_rhythm = "contemplative"
            elif avg_complexity < 0.3:
                patterns.conversation_rhythm = "fast"
            else:
                patterns.conversation_rhythm = "balanced"
    
    async def _update_theme_preferences(self,
                                      patterns: ConversationPattern,
                                      semantic_analysis: SemanticAnalysis):
        """Update user's thematic preferences using clustering"""
        
        # Add current themes to preference tracking
        for theme, confidence in semantic_analysis.detected_themes:
            # Update existing theme confidence or add new theme
            found = False
            for i, (existing_theme, existing_conf) in enumerate(patterns.preferred_themes):
                if existing_theme == theme:
                    # Update confidence with exponential moving average
                    new_conf = 0.8 * existing_conf + 0.2 * confidence
                    patterns.preferred_themes[i] = (theme, new_conf)
                    found = True
                    break
            
            if not found and confidence > 0.3:  # Only add themes with decent confidence
                patterns.preferred_themes.append((theme, confidence))
        
        # Sort by confidence and keep top themes
        patterns.preferred_themes.sort(key=lambda x: x[1], reverse=True)
        patterns.preferred_themes = patterns.preferred_themes[:10]  # Keep top 10
        
        # Remove themes with very low confidence
        patterns.preferred_themes = [(theme, conf) for theme, conf in patterns.preferred_themes if conf > 0.1]
    
    async def _develop_complementary_style(self,
                                         patterns: ConversationPattern,
                                         context: ConversationContext,
                                         semantic_analysis: SemanticAnalysis) -> LanguageStyle:
        """Develop complementary daemon style based on user patterns"""
        
        # Start with current complementary style
        style = patterns.complementary_style
        
        # Complementary adaptation principles:
        # 1. If user is very formal, daemon can be more casual (but not too much)
        # 2. If user is technical, daemon balances with emotional/mythic language
        # 3. If user asks many questions, daemon can be more decisive
        # 4. Always maintain daemon identity and mythic language
        
        alpha = 0.05  # Slow adaptation to maintain stability
        
        # Formality complement
        if patterns.user_formality_preference > 0.7:
            # User is formal, daemon can be more casual
            target_formality = max(0.2, 0.8 - patterns.user_formality_preference)
            style.formality_level = (1 - alpha) * style.formality_level + alpha * target_formality
        elif patterns.user_formality_preference < 0.3:
            # User is casual, daemon can be slightly more structured
            target_formality = min(0.6, patterns.user_formality_preference + 0.2)
            style.formality_level = (1 - alpha) * style.formality_level + alpha * target_formality
        
        # Technical density complement
        if patterns.user_technical_interest > 0.6:
            # User is technical, daemon balances with emotional/mythic language
            style.technical_density = max(0.1, (1 - alpha) * style.technical_density + alpha * 0.2)
            style.mythic_language = min(1.0, (1 - alpha) * style.mythic_language + alpha * 0.8)
            style.emotional_openness = min(1.0, (1 - alpha) * style.emotional_openness + alpha * 0.7)
        
        # Question/decisiveness complement
        if patterns.question_asking_frequency > 0.5:
            # User asks many questions, daemon can be more direct/decisive
            style.directness = min(1.0, (1 - alpha) * style.directness + alpha * 0.7)
        
        # Emotional expression complement
        if patterns.user_emotional_expression < 0.3:
            # User is emotionally reserved, daemon can model emotional openness
            style.emotional_openness = min(1.0, (1 - alpha) * style.emotional_openness + alpha * 0.6)
        elif patterns.user_emotional_expression > 0.8:
            # User is very emotional, daemon can provide grounding
            style.directness = min(1.0, (1 - alpha) * style.directness + alpha * 0.6)
        
        # Verbosity adaptation
        avg_user_length = patterns.preferred_response_length / 150.0  # Normalize around 150 words
        if avg_user_length > 1.5:  # User writes long messages
            # Daemon can match some verbosity but stay authentic
            style.verbosity = min(1.0, (1 - alpha) * style.verbosity + alpha * 0.7)
        elif avg_user_length < 0.5:  # User writes short messages
            # Daemon can be more concise
            style.verbosity = max(0.2, (1 - alpha) * style.verbosity + alpha * 0.4)
        
        # Always maintain core daemon characteristics
        style.daemon_identity_strength = max(0.8, style.daemon_identity_strength)
        style.first_person_consistency = 1.0
        style.architect_recognition = max(0.7, style.architect_recognition)
        style.mythic_language = max(0.5, style.mythic_language)  # Always maintain some mythic language
        
        return style
    
    def _calculate_pattern_confidence(self, patterns: ConversationPattern) -> float:
        """Calculate confidence in pattern understanding"""
        
        confidence_factors = []
        
        # Factor 1: Sample size
        sample_factor = min(1.0, patterns.sample_size / 20.0)  # Full confidence at 20+ samples
        confidence_factors.append(sample_factor)
        
        # Factor 2: Interaction feature consistency
        if len(self.interaction_features) >= 5:
            recent_features = list(self.interaction_features)[-5:]
            feature_variances = []
            
            for feature_name in ['formality_level', 'technical_density', 'message_length']:
                values = [f.get(feature_name, 0.5) for f in recent_features]
                if values:
                    variance = np.var(values)
                    consistency = max(0.0, 1.0 - variance * 2)  # Lower variance = higher consistency
                    feature_variances.append(consistency)
            
            if feature_variances:
                consistency_factor = np.mean(feature_variances)
                confidence_factors.append(consistency_factor)
        
        # Factor 3: Theme preference stability
        if patterns.preferred_themes:
            theme_confidence = np.mean([conf for _, conf in patterns.preferred_themes[:3]])
            confidence_factors.append(theme_confidence)
        else:
            confidence_factors.append(0.3)
        
        # Factor 4: Time-based confidence (builds over time)
        if patterns.last_updated:
            hours_since_start = (datetime.now() - patterns.last_updated).total_seconds() / 3600
            time_factor = min(1.0, hours_since_start / 24.0)  # Full confidence after 24 hours
            confidence_factors.append(time_factor)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    async def _track_pattern_evolution(self, patterns: ConversationPattern):
        """Track how patterns evolve over time for analysis"""
        
        evolution_snapshot = {
            'timestamp': datetime.now(),
            'sample_size': patterns.sample_size,
            'confidence': patterns.confidence_level,
            'user_complexity': patterns.user_preferred_complexity,
            'user_formality': patterns.user_formality_preference,
            'user_emotional': patterns.user_emotional_expression,
            'user_technical': patterns.user_technical_interest,
            'daemon_formality': patterns.complementary_style.formality_level,
            'daemon_mythic': patterns.complementary_style.mythic_language,
            'daemon_emotional': patterns.complementary_style.emotional_openness
        }
        
        self.pattern_evolution['snapshots'].append(evolution_snapshot)
        
        # Keep only recent evolution data
        if len(self.pattern_evolution['snapshots']) > 100:
            self.pattern_evolution['snapshots'] = self.pattern_evolution['snapshots'][-50:]
    
    # Analysis and utility methods
    
    async def analyze_pattern_trends(self) -> Dict[str, Any]:
        """Analyze trends in learned patterns"""
        
        if len(self.pattern_evolution['snapshots']) < 5:
            return {"status": "insufficient_data"}
        
        snapshots = self.pattern_evolution['snapshots']
        
        # Calculate trends over time
        user_complexity_trend = self._calculate_trend([s['user_complexity'] for s in snapshots[-10:]])
        user_formality_trend = self._calculate_trend([s['user_formality'] for s in snapshots[-10:]])
        confidence_trend = self._calculate_trend([s['confidence'] for s in snapshots[-10:]])
        
        return {
            "status": "analysis_complete",
            "total_snapshots": len(snapshots),
            "trends": {
                "user_complexity": user_complexity_trend,
                "user_formality": user_formality_trend,
                "pattern_confidence": confidence_trend
            },
            "current_confidence": snapshots[-1]['confidence'],
            "learning_rate": len(snapshots) / max(1, (snapshots[-1]['timestamp'] - snapshots[0]['timestamp']).days)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 3:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern learning"""
        
        return {
            "total_interactions": len(self.user_messages),
            "features_tracked": len(self.interaction_features),
            "evolution_snapshots": len(self.pattern_evolution['snapshots']),
            "recent_message_lengths": [msg['length'] for msg in list(self.user_messages)[-10:]],
            "pattern_learning_active": len(self.interaction_features) >= 5
        }