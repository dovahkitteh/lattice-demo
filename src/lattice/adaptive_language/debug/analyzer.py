"""
Conversation Analyzer

Deep analysis tools for understanding conversation patterns, user behavior,
and system adaptation effectiveness in the adaptive language system.
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
except ImportError:
    KMeans = None
    cosine_similarity = None
    PCA = None

from ..core.models import ConversationContext, SemanticAnalysis, MoodState, LanguageStyle

logger = logging.getLogger(__name__)


class ConversationAnalyzer:
    """
    Advanced conversation analysis for understanding user patterns,
    system adaptation effectiveness, and conversation dynamics.
    """
    
    def __init__(self):
        self.conversation_data = []
        self.user_profiles = {}
        self.adaptation_history = []
        
        logger.info("📊 Conversation Analyzer initialized")
    
    def add_conversation_turn(self,
                            user_message: str,
                            daemon_response: str,
                            context: ConversationContext,
                            semantic_analysis: SemanticAnalysis,
                            mood_state: MoodState,
                            language_style: LanguageStyle,
                            session_id: Optional[str] = None):
        """Add a conversation turn for analysis"""
        
        turn_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or 'unknown',
            'user_message': {
                'text': user_message,
                'word_count': len(user_message.split()),
                'char_count': len(user_message),
                'contains_questions': '?' in user_message,
                'semantic_analysis': {
                    'intent': semantic_analysis.intent_classification,
                    'themes': semantic_analysis.detected_themes,
                    'emotional_subtext': semantic_analysis.emotional_subtext,
                    'formality': semantic_analysis.formality_level,
                    'technical_density': semantic_analysis.technical_density,
                    'confidence': semantic_analysis.analysis_confidence
                }
            },
            'daemon_response': {
                'text': daemon_response,
                'word_count': len(daemon_response.split()),
                'char_count': len(daemon_response)
            },
            'context': {
                'conversation_length': context.conversation_length,
                'energy_level': context.energy_level,
                'intimacy_level': context.intimacy_level,
                'stagnancy_risk': context.stagnancy_risk,
                'evolution_pressure': context.evolution_pressure
            },
            'mood_state': {
                'spectrum': mood_state.spectrum_position.value,
                'lightness': mood_state.lightness,
                'engagement': mood_state.engagement,
                'profundity': mood_state.profundity,
                'warmth': mood_state.warmth,
                'intensity': mood_state.intensity
            },
            'language_style': {
                'formality': language_style.formality_level,
                'technical_density': language_style.technical_density,
                'mythic_language': language_style.mythic_language,
                'directness': language_style.directness,
                'verbosity': language_style.verbosity,
                'emotional_openness': language_style.emotional_openness
            }
        }
        
        self.conversation_data.append(turn_data)
        
        # Limit memory usage
        if len(self.conversation_data) > 2000:
            self.conversation_data = self.conversation_data[-1500:]
    
    def analyze_user_patterns(self, session_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Analyze user communication patterns"""
        
        # Filter data
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_data = [
            turn for turn in self.conversation_data
            if (datetime.fromisoformat(turn['timestamp']) > cutoff_time and
                (session_id is None or turn['session_id'] == session_id))
        ]
        
        if not filtered_data:
            return {"status": "no_data", "period_hours": hours}
        
        # Extract user message patterns
        user_messages = [turn['user_message'] for turn in filtered_data]
        
        analysis = {
            'period_hours': hours,
            'total_turns': len(filtered_data),
            'communication_style': self._analyze_communication_style(user_messages),
            'content_patterns': self._analyze_content_patterns(user_messages),
            'interaction_dynamics': self._analyze_interaction_dynamics(filtered_data),
            'temporal_patterns': self._analyze_temporal_patterns(filtered_data),
            'emotional_patterns': self._analyze_emotional_patterns(user_messages)
        }
        
        return analysis
    
    def analyze_adaptation_effectiveness(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze how effectively the system adapts to users"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            turn for turn in self.conversation_data
            if datetime.fromisoformat(turn['timestamp']) > cutoff_time
        ]
        
        if not recent_data:
            return {"status": "no_data", "period_hours": hours}
        
        # Group by session for adaptation analysis
        sessions = defaultdict(list)
        for turn in recent_data:
            sessions[turn['session_id']].append(turn)
        
        adaptation_metrics = {
            'period_hours': hours,
            'sessions_analyzed': len(sessions),
            'total_turns': len(recent_data),
            'adaptation_trends': {},
            'effectiveness_scores': {},
            'session_summaries': {}
        }
        
        for session_id, turns in sessions.items():
            if len(turns) < 3:  # Need enough turns for trend analysis
                continue
            
            session_analysis = self._analyze_session_adaptation(turns)
            adaptation_metrics['session_summaries'][session_id] = session_analysis
        
        # Calculate overall adaptation effectiveness
        if adaptation_metrics['session_summaries']:
            session_scores = [s['adaptation_score'] for s in adaptation_metrics['session_summaries'].values()]
            adaptation_metrics['overall_adaptation_score'] = sum(session_scores) / len(session_scores)
            
            # Analyze trends across sessions
            adaptation_metrics['adaptation_trends'] = self._calculate_adaptation_trends(
                list(adaptation_metrics['session_summaries'].values())
            )
        
        return adaptation_metrics
    
    def analyze_conversation_quality(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze overall conversation quality metrics"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            turn for turn in self.conversation_data
            if datetime.fromisoformat(turn['timestamp']) > cutoff_time
        ]
        
        if not recent_data:
            return {"status": "no_data", "period_hours": hours}
        
        quality_metrics = {
            'period_hours': hours,
            'total_turns': len(recent_data),
            'semantic_quality': self._analyze_semantic_quality(recent_data),
            'mood_quality': self._analyze_mood_quality(recent_data),
            'response_quality': self._analyze_response_quality(recent_data),
            'engagement_quality': self._analyze_engagement_quality(recent_data),
            'overall_quality_score': 0.0
        }
        
        # Calculate overall quality score
        quality_scores = [
            quality_metrics['semantic_quality']['score'],
            quality_metrics['mood_quality']['score'],
            quality_metrics['response_quality']['score'],
            quality_metrics['engagement_quality']['score']
        ]
        
        quality_metrics['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return quality_metrics
    
    def _analyze_communication_style(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Analyze user's communication style patterns"""
        
        if not user_messages:
            return {}
        
        # Calculate style metrics
        avg_length = sum(msg['word_count'] for msg in user_messages) / len(user_messages)
        question_rate = sum(1 for msg in user_messages if msg['contains_questions']) / len(user_messages)
        
        formality_scores = [msg['semantic_analysis']['formality'] for msg in user_messages 
                          if msg['semantic_analysis']['formality'] is not None]
        avg_formality = sum(formality_scores) / len(formality_scores) if formality_scores else 0.5
        
        technical_scores = [msg['semantic_analysis']['technical_density'] for msg in user_messages
                          if msg['semantic_analysis']['technical_density'] is not None]
        avg_technical = sum(technical_scores) / len(technical_scores) if technical_scores else 0.0
        
        return {
            'average_message_length': avg_length,
            'question_frequency': question_rate,
            'formality_level': avg_formality,
            'technical_density': avg_technical,
            'style_classification': self._classify_communication_style(avg_length, question_rate, avg_formality),
            'consistency_score': self._calculate_style_consistency(user_messages)
        }
    
    def _analyze_content_patterns(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Analyze content patterns in user messages"""
        
        # Theme analysis
        all_themes = []
        intent_counts = Counter()
        emotional_subtexts = Counter()
        
        for msg in user_messages:
            analysis = msg['semantic_analysis']
            all_themes.extend([theme for theme, conf in analysis['themes']])
            intent_counts[analysis['intent']] += 1
            emotional_subtexts[analysis['emotional_subtext']] += 1
        
        theme_counts = Counter(all_themes)
        
        return {
            'most_common_themes': theme_counts.most_common(5),
            'intent_distribution': dict(intent_counts),
            'emotional_tone_distribution': dict(emotional_subtexts),
            'theme_diversity': len(theme_counts) / max(len(all_themes), 1),
            'dominant_intent': intent_counts.most_common(1)[0] if intent_counts else None
        }
    
    def _analyze_interaction_dynamics(self, conversation_turns: List[Dict]) -> Dict[str, Any]:
        """Analyze interaction dynamics between user and daemon"""
        
        if len(conversation_turns) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate response ratios
        user_lengths = [turn['user_message']['word_count'] for turn in conversation_turns]
        daemon_lengths = [turn['daemon_response']['word_count'] for turn in conversation_turns]
        
        avg_user_length = sum(user_lengths) / len(user_lengths)
        avg_daemon_length = sum(daemon_lengths) / len(daemon_lengths)
        
        # Analyze conversation flow
        energy_changes = []
        intimacy_changes = []
        
        for i in range(1, len(conversation_turns)):
            prev_context = conversation_turns[i-1]['context']
            curr_context = conversation_turns[i]['context']
            
            energy_changes.append(curr_context['energy_level'] - prev_context['energy_level'])
            intimacy_changes.append(curr_context['intimacy_level'] - prev_context['intimacy_level'])
        
        return {
            'response_length_ratio': avg_daemon_length / max(avg_user_length, 1),
            'energy_trend': 'increasing' if sum(energy_changes) > 0 else 'decreasing',
            'intimacy_trend': 'increasing' if sum(intimacy_changes) > 0 else 'decreasing',
            'conversation_flow_score': self._calculate_flow_score(conversation_turns),
            'turn_consistency': self._calculate_turn_consistency(conversation_turns)
        }
    
    def _analyze_temporal_patterns(self, conversation_turns: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in conversation"""
        
        timestamps = [datetime.fromisoformat(turn['timestamp']) for turn in conversation_turns]
        
        if len(timestamps) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate time intervals
        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                    for i in range(1, len(timestamps))]
        
        return {
            'conversation_duration_minutes': (timestamps[-1] - timestamps[0]).total_seconds() / 60,
            'average_response_interval_seconds': sum(intervals) / len(intervals),
            'conversation_pace': 'fast' if sum(intervals) / len(intervals) < 60 else 'slow',
            'peak_activity_hours': self._find_peak_hours(timestamps)
        }
    
    def _analyze_emotional_patterns(self, user_messages: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns in user communication"""
        
        emotional_subtexts = [msg['semantic_analysis']['emotional_subtext'] for msg in user_messages]
        
        # Map emotions to valence
        emotion_valence = {
            'very positive': 1.0,
            'positive': 0.5,
            'neutral': 0.0,
            'negative': -0.5,
            'very negative': -1.0
        }
        
        valences = [emotion_valence.get(emotion, 0.0) for emotion in emotional_subtexts]
        
        return {
            'emotional_distribution': Counter(emotional_subtexts),
            'average_valence': sum(valences) / len(valences) if valences else 0.0,
            'emotional_stability': 1.0 - (np.std(valences) if len(valences) > 1 else 0.0),
            'dominant_emotion': Counter(emotional_subtexts).most_common(1)[0] if emotional_subtexts else None
        }
    
    def _analyze_session_adaptation(self, session_turns: List[Dict]) -> Dict[str, Any]:
        """Analyze adaptation within a single session"""
        
        if len(session_turns) < 3:
            return {"adaptation_score": 0.5, "status": "insufficient_data"}
        
        # Track adaptation metrics over the session
        language_styles = [turn['language_style'] for turn in session_turns]
        mood_states = [turn['mood_state'] for turn in session_turns]
        
        # Calculate adaptation score based on style changes
        style_changes = 0
        for i in range(1, len(language_styles)):
            prev_style = language_styles[i-1]
            curr_style = language_styles[i]
            
            # Count significant changes
            for key in ['formality', 'directness', 'verbosity']:
                if abs(curr_style[key] - prev_style[key]) > 0.1:
                    style_changes += 1
        
        # Mood adaptation analysis
        mood_changes = 0
        for i in range(1, len(mood_states)):
            prev_mood = mood_states[i-1]
            curr_mood = mood_states[i]
            
            if prev_mood['spectrum'] != curr_mood['spectrum']:
                mood_changes += 1
        
        adaptation_score = min(1.0, (style_changes + mood_changes) / len(session_turns))
        
        return {
            'adaptation_score': adaptation_score,
            'style_changes': style_changes,
            'mood_changes': mood_changes,
            'final_mood': mood_states[-1]['spectrum'],
            'session_length': len(session_turns),
            'adaptation_rate': adaptation_score / len(session_turns)
        }
    
    def _calculate_adaptation_trends(self, session_analyses: List[Dict]) -> Dict[str, Any]:
        """Calculate trends across multiple sessions"""
        
        adaptation_scores = [s['adaptation_score'] for s in session_analyses]
        mood_diversity = len(set(s['final_mood'] for s in session_analyses)) / len(session_analyses)
        
        return {
            'average_adaptation_score': sum(adaptation_scores) / len(adaptation_scores),
            'adaptation_consistency': 1.0 - np.std(adaptation_scores),
            'mood_diversity': mood_diversity,
            'trend_direction': 'improving' if adaptation_scores[-1] > adaptation_scores[0] else 'stable'
        }
    
    def _analyze_semantic_quality(self, turns: List[Dict]) -> Dict[str, Any]:
        """Analyze semantic analysis quality"""
        
        confidences = [turn['user_message']['semantic_analysis']['confidence'] for turn in turns]
        
        return {
            'score': sum(confidences) / len(confidences),
            'consistency': 1.0 - np.std(confidences),
            'high_confidence_rate': sum(1 for c in confidences if c > 0.7) / len(confidences)
        }
    
    def _analyze_mood_quality(self, turns: List[Dict]) -> Dict[str, Any]:
        """Analyze mood detection quality"""
        
        mood_spectrums = [turn['mood_state']['spectrum'] for turn in turns]
        mood_variety = len(set(mood_spectrums)) / 3  # 3 possible spectrums
        
        return {
            'score': mood_variety,
            'spectrum_distribution': Counter(mood_spectrums),
            'transition_smoothness': self._calculate_mood_transition_smoothness(turns)
        }
    
    def _analyze_response_quality(self, turns: List[Dict]) -> Dict[str, Any]:
        """Analyze daemon response quality"""
        
        response_lengths = [turn['daemon_response']['word_count'] for turn in turns]
        avg_length = sum(response_lengths) / len(response_lengths)
        
        # Quality based on appropriate response length and variety
        length_consistency = 1.0 - min(1.0, np.std(response_lengths) / max(avg_length, 1))
        
        return {
            'score': min(1.0, avg_length / 50) * length_consistency,  # Normalize around 50 words
            'average_length': avg_length,
            'length_consistency': length_consistency
        }
    
    def _analyze_engagement_quality(self, turns: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation engagement quality"""
        
        energy_levels = [turn['context']['energy_level'] for turn in turns]
        intimacy_levels = [turn['context']['intimacy_level'] for turn in turns]
        
        avg_energy = sum(energy_levels) / len(energy_levels)
        avg_intimacy = sum(intimacy_levels) / len(intimacy_levels)
        
        return {
            'score': (avg_energy + avg_intimacy) / 2,
            'energy_level': avg_energy,
            'intimacy_level': avg_intimacy,
            'engagement_trend': 'increasing' if energy_levels[-1] > energy_levels[0] else 'stable'
        }
    
    # Helper methods
    
    def _classify_communication_style(self, avg_length: float, question_rate: float, formality: float) -> str:
        """Classify user communication style"""
        
        if avg_length > 20 and formality > 0.6:
            return "formal_detailed"
        elif question_rate > 0.5:
            return "inquisitive"
        elif avg_length < 10:
            return "concise"
        elif formality < 0.3:
            return "casual"
        else:
            return "balanced"
    
    def _calculate_style_consistency(self, user_messages: List[Dict]) -> float:
        """Calculate consistency in user communication style"""
        
        if len(user_messages) < 2:
            return 1.0
        
        lengths = [msg['word_count'] for msg in user_messages]
        formalities = [msg['semantic_analysis']['formality'] for msg in user_messages 
                      if msg['semantic_analysis']['formality'] is not None]
        
        length_consistency = 1.0 - min(1.0, np.std(lengths) / max(np.mean(lengths), 1))
        formality_consistency = 1.0 - np.std(formalities) if len(formalities) > 1 else 1.0
        
        return (length_consistency + formality_consistency) / 2
    
    def _calculate_flow_score(self, turns: List[Dict]) -> float:
        """Calculate conversation flow quality score"""
        
        if len(turns) < 2:
            return 0.5
        
        # Measure flow based on energy and context continuity
        flow_factors = []
        
        for i in range(1, len(turns)):
            prev_turn = turns[i-1]
            curr_turn = turns[i]
            
            # Energy continuity
            energy_diff = abs(curr_turn['context']['energy_level'] - prev_turn['context']['energy_level'])
            energy_factor = max(0.0, 1.0 - energy_diff)
            
            # Mood appropriateness
            mood_factor = 0.8  # Placeholder - could be more sophisticated
            
            flow_factors.append((energy_factor + mood_factor) / 2)
        
        return sum(flow_factors) / len(flow_factors)
    
    def _calculate_turn_consistency(self, turns: List[Dict]) -> float:
        """Calculate consistency across conversation turns"""
        
        # Measure consistency in response styles and moods
        daemon_lengths = [turn['daemon_response']['word_count'] for turn in turns]
        mood_changes = sum(1 for i in range(1, len(turns)) 
                          if turns[i]['mood_state']['spectrum'] != turns[i-1]['mood_state']['spectrum'])
        
        length_consistency = 1.0 - min(1.0, np.std(daemon_lengths) / max(np.mean(daemon_lengths), 1))
        mood_consistency = 1.0 - min(1.0, mood_changes / len(turns))
        
        return (length_consistency + mood_consistency) / 2
    
    def _find_peak_hours(self, timestamps: List[datetime]) -> List[int]:
        """Find peak conversation hours"""
        
        hours = [ts.hour for ts in timestamps]
        hour_counts = Counter(hours)
        
        if not hour_counts:
            return []
        
        max_count = max(hour_counts.values())
        return [hour for hour, count in hour_counts.items() if count == max_count]
    
    def _calculate_mood_transition_smoothness(self, turns: List[Dict]) -> float:
        """Calculate smoothness of mood transitions"""
        
        if len(turns) < 2:
            return 1.0
        
        # Measure how gradually moods change
        mood_dimensions = ['lightness', 'engagement', 'profundity', 'warmth', 'intensity']
        
        transition_scores = []
        for i in range(1, len(turns)):
            prev_mood = turns[i-1]['mood_state']
            curr_mood = turns[i]['mood_state']
            
            # Calculate Euclidean distance between mood states
            distance = np.sqrt(sum((curr_mood[dim] - prev_mood[dim])**2 for dim in mood_dimensions))
            smoothness = max(0.0, 1.0 - distance)  # Closer = smoother
            transition_scores.append(smoothness)
        
        return sum(transition_scores) / len(transition_scores)
    
    def export_analysis_report(self, filename: Optional[str] = None) -> Path:
        """Export comprehensive analysis report"""
        
        if filename is None:
            filename = f"conversation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_path = Path("data/adaptive_language_logs") / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_turns': len(self.conversation_data),
                'date_range': {
                    'start': self.conversation_data[0]['timestamp'] if self.conversation_data else None,
                    'end': self.conversation_data[-1]['timestamp'] if self.conversation_data else None
                }
            },
            'user_patterns': self.analyze_user_patterns(),
            'adaptation_effectiveness': self.analyze_adaptation_effectiveness(),
            'conversation_quality': self.analyze_conversation_quality(),
            'raw_data_sample': self.conversation_data[-10:] if self.conversation_data else []
        }
        
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Analysis report exported to {export_path}")
        return export_path


# Global analyzer instance
_global_analyzer = None

def get_conversation_analyzer() -> ConversationAnalyzer:
    """Get global conversation analyzer instance"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ConversationAnalyzer()
    return _global_analyzer