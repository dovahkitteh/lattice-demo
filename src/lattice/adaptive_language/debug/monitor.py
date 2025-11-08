"""
Language System Monitor

Real-time monitoring and debugging tools for the adaptive language system.
Provides insights into mood detection, pattern learning, and system performance.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from pathlib import Path

from ..core.models import MoodState, ConversationContext, SemanticAnalysis, LanguageStyle

logger = logging.getLogger(__name__)


class LanguageSystemMonitor:
    """
    Comprehensive monitoring system for adaptive language components
    
    Tracks mood transitions, semantic analysis quality, pattern learning progress,
    and system performance metrics for debugging and optimization.
    """
    
    def __init__(self, log_directory: Optional[Path] = None):
        self.log_directory = log_directory or Path("data/adaptive_language_logs")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Real-time monitoring data
        self.mood_transitions = deque(maxlen=1000)
        self.semantic_analyses = deque(maxlen=500)
        self.performance_metrics = deque(maxlen=200)
        self.pattern_learning_history = deque(maxlen=100)
        
        # System health tracking
        self.component_errors = defaultdict(list)
        self.warning_counts = defaultdict(int)
        self.last_health_check = datetime.now()
        
        # Session tracking
        self.current_session = {
            'start_time': datetime.now(),
            'interaction_count': 0,
            'mood_changes': 0,
            'semantic_confidence_sum': 0.0,
            'average_processing_time': 0.0
        }
        
        logger.info("🔍 Language System Monitor initialized")
    
    def log_mood_transition(self, 
                           previous_mood: MoodState,
                           new_mood: MoodState,
                           context: ConversationContext,
                           semantic_analysis: SemanticAnalysis):
        """Log a mood state transition with context"""
        
        transition_data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': getattr(context, 'session_id', 'unknown'),
            'message_preview': context.user_message[:100],
            'previous_mood': {
                'spectrum': previous_mood.spectrum_position.value,
                'coordinates': previous_mood.mood_vector.tolist(),
                'stability': previous_mood.stability
            },
            'new_mood': {
                'spectrum': new_mood.spectrum_position.value,
                'coordinates': new_mood.mood_vector.tolist(),
                'stability': new_mood.stability
            },
            'context_factors': {
                'stagnancy_risk': context.stagnancy_risk,
                'evolution_pressure': context.evolution_pressure,
                'energy_level': context.energy_level,
                'conversation_length': context.conversation_length
            },
            'semantic_triggers': {
                'intent': semantic_analysis.intent_classification,
                'themes': [theme for theme, conf in semantic_analysis.detected_themes[:3]],
                'emotional_subtext': semantic_analysis.emotional_subtext,
                'confidence': semantic_analysis.analysis_confidence
            },
            'mood_distance': previous_mood.distance_to(new_mood)
        }
        
        self.mood_transitions.append(transition_data)
        self.current_session['mood_changes'] += 1
        
        # Log significant mood shifts
        if transition_data['mood_distance'] > 0.5:
            logger.info(f"🎭 MONITOR: Significant mood shift - {previous_mood.spectrum_position.value} → {new_mood.spectrum_position.value} (distance: {transition_data['mood_distance']:.2f})")
    
    def log_semantic_analysis(self,
                             message: str,
                             analysis: SemanticAnalysis,
                             processing_time_ms: int):
        """Log semantic analysis results and quality metrics"""
        
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'message_length': len(message.split()),
            'message_preview': message[:100],
            'analysis': {
                'intent': analysis.intent_classification,
                'confidence': analysis.analysis_confidence,
                'themes': analysis.detected_themes,
                'emotional_subtext': analysis.emotional_subtext,
                'contains_questions': analysis.contains_questions,
                'contains_challenges': analysis.contains_challenges,
                'contains_paradoxes': analysis.contains_paradoxes,
                'formality_level': analysis.formality_level,
                'technical_density': analysis.technical_density,
                'creative_language': analysis.creative_language
            },
            'processing_time_ms': processing_time_ms,
            'quality_indicators': {
                'high_confidence': analysis.analysis_confidence > 0.7,
                'themes_detected': len(analysis.detected_themes) > 0,
                'intent_classified': analysis.intent_classification != "general_conversation",
                'fast_processing': processing_time_ms < 1000
            }
        }
        
        self.semantic_analyses.append(analysis_data)
        self.current_session['semantic_confidence_sum'] += analysis.analysis_confidence
        
        # Track analysis quality
        if analysis.analysis_confidence < 0.3:
            self.warning_counts['low_semantic_confidence'] += 1
            logger.warning(f"🧠 MONITOR: Low semantic analysis confidence: {analysis.analysis_confidence:.2f}")
    
    def log_performance_metrics(self,
                               component: str,
                               operation: str,
                               duration_ms: int,
                               success: bool,
                               details: Optional[Dict] = None):
        """Log performance metrics for system components"""
        
        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'details': details or {}
        }
        
        self.performance_metrics.append(metric_data)
        
        # Track slow operations
        slow_thresholds = {
            'semantic_analysis': 2000,
            'mood_detection': 500,
            'prompt_building': 1000,
            'pattern_learning': 300
        }
        
        threshold = slow_thresholds.get(component, 1000)
        if duration_ms > threshold:
            self.warning_counts['slow_operation'] += 1
            logger.warning(f"⚡ MONITOR: Slow {component} operation: {duration_ms}ms ({operation})")
        
        # Track failures
        if not success:
            self.component_errors[component].append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'details': details
            })
            logger.error(f"❌ MONITOR: Component failure - {component}.{operation}")
    
    def log_pattern_learning_update(self,
                                  patterns_before: Dict[str, float],
                                  patterns_after: Dict[str, float],
                                  confidence_change: float,
                                  sample_size: int):
        """Log pattern learning progress"""
        
        learning_data = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size,
            'confidence_change': confidence_change,
            'pattern_changes': {},
            'learning_rate': confidence_change / max(sample_size, 1)
        }
        
        # Calculate pattern changes
        for key in patterns_after:
            if key in patterns_before:
                change = patterns_after[key] - patterns_before[key]
                if abs(change) > 0.01:  # Only log significant changes
                    learning_data['pattern_changes'][key] = {
                        'before': patterns_before[key],
                        'after': patterns_after[key],
                        'change': change
                    }
        
        self.pattern_learning_history.append(learning_data)
        
        # Log significant learning events
        if confidence_change > 0.1:
            logger.info(f"🧠 MONITOR: Significant pattern learning - confidence change: {confidence_change:.3f}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        now = datetime.now()
        session_duration = (now - self.current_session['start_time']).total_seconds()
        
        health_report = {
            'timestamp': now.isoformat(),
            'session_duration_minutes': session_duration / 60,
            'overall_status': 'healthy',
            'current_session': self.current_session.copy(),
            'component_health': {},
            'performance_summary': {},
            'warning_summary': dict(self.warning_counts),
            'recent_errors': {
                component: errors[-5:] for component, errors in self.component_errors.items()
            }
        }
        
        # Calculate session averages
        if self.current_session['interaction_count'] > 0:
            health_report['current_session']['avg_semantic_confidence'] = (
                self.current_session['semantic_confidence_sum'] / self.current_session['interaction_count']
            )
        
        # Component health analysis
        for component in ['semantic_analysis', 'mood_detection', 'prompt_building', 'pattern_learning']:
            recent_metrics = [m for m in self.performance_metrics 
                            if m['component'] == component and 
                            datetime.fromisoformat(m['timestamp']) > now - timedelta(minutes=30)]
            
            if recent_metrics:
                successes = sum(1 for m in recent_metrics if m['success'])
                avg_duration = sum(m['duration_ms'] for m in recent_metrics) / len(recent_metrics)
                
                health_report['component_health'][component] = {
                    'success_rate': successes / len(recent_metrics),
                    'avg_duration_ms': avg_duration,
                    'recent_operations': len(recent_metrics),
                    'status': 'healthy' if successes / len(recent_metrics) > 0.9 else 'degraded'
                }
        
        # Performance summary
        if self.performance_metrics:
            recent_metrics = [m for m in self.performance_metrics 
                            if datetime.fromisoformat(m['timestamp']) > now - timedelta(minutes=10)]
            
            if recent_metrics:
                health_report['performance_summary'] = {
                    'total_operations': len(recent_metrics),
                    'avg_duration_ms': sum(m['duration_ms'] for m in recent_metrics) / len(recent_metrics),
                    'success_rate': sum(1 for m in recent_metrics if m['success']) / len(recent_metrics),
                    'operations_per_minute': len(recent_metrics) / 10
                }
        
        # Determine overall status
        error_rate = sum(len(errors) for errors in self.component_errors.values()) / max(session_duration / 60, 1)
        warning_rate = sum(self.warning_counts.values()) / max(session_duration / 60, 1)
        
        if error_rate > 5 or warning_rate > 10:
            health_report['overall_status'] = 'unhealthy'
        elif error_rate > 1 or warning_rate > 5:
            health_report['overall_status'] = 'degraded'
        
        return health_report
    
    def get_mood_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze mood patterns over specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_transitions = [
            t for t in self.mood_transitions 
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]
        
        if not recent_transitions:
            return {"status": "no_data", "period_hours": hours}
        
        # Mood distribution analysis
        mood_counts = defaultdict(int)
        mood_distances = []
        stagnancy_events = 0
        
        for transition in recent_transitions:
            mood_counts[transition['new_mood']['spectrum']] += 1
            mood_distances.append(transition['mood_distance'])
            
            if transition['context_factors']['stagnancy_risk'] > 0.7:
                stagnancy_events += 1
        
        # Calculate mood variety
        total_transitions = len(recent_transitions)
        mood_variety = len(mood_counts) / 3  # Normalize by 3 possible spectrums
        
        analysis = {
            'period_hours': hours,
            'total_transitions': total_transitions,
            'mood_distribution': dict(mood_counts),
            'mood_variety_score': mood_variety,
            'avg_mood_distance': sum(mood_distances) / len(mood_distances),
            'stagnancy_events': stagnancy_events,
            'stagnancy_rate': stagnancy_events / total_transitions,
            'most_common_mood': max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else None,
            'transition_frequency': total_transitions / hours if hours > 0 else 0
        }
        
        # Health assessment
        if mood_variety < 0.5:
            analysis['mood_health'] = 'low_variety'
        elif analysis['stagnancy_rate'] > 0.3:
            analysis['mood_health'] = 'high_stagnancy'
        elif analysis['avg_mood_distance'] < 0.1:
            analysis['mood_health'] = 'minimal_change'
        else:
            analysis['mood_health'] = 'healthy'
        
        return analysis
    
    def export_session_log(self, filename: Optional[str] = None) -> Path:
        """Export comprehensive session data for analysis"""
        
        if filename is None:
            filename = f"language_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_path = self.log_directory / filename
        
        export_data = {
            'session_info': self.current_session,
            'system_health': self.get_system_health(),
            'mood_analysis': self.get_mood_analysis(),
            'mood_transitions': list(self.mood_transitions),
            'semantic_analyses': list(self.semantic_analyses),
            'performance_metrics': list(self.performance_metrics),
            'pattern_learning_history': list(self.pattern_learning_history),
            'component_errors': dict(self.component_errors),
            'warning_counts': dict(self.warning_counts)
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"📊 MONITOR: Session log exported to {export_path}")
        return export_path
    
    def reset_session(self):
        """Reset session tracking for new session"""
        
        # Export current session before reset
        self.export_session_log()
        
        # Reset session data
        self.current_session = {
            'start_time': datetime.now(),
            'interaction_count': 0,
            'mood_changes': 0,
            'semantic_confidence_sum': 0.0,
            'average_processing_time': 0.0
        }
        
        # Clear warning counts but keep some history
        self.warning_counts.clear()
        
        logger.info("🔄 MONITOR: Session reset, previous session exported")
    
    def get_debug_insights(self) -> Dict[str, Any]:
        """Get debugging insights and recommendations"""
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'observations': [],
            'potential_issues': []
        }
        
        # Analyze mood patterns
        mood_analysis = self.get_mood_analysis(hours=2)
        if mood_analysis.get('mood_health') == 'low_variety':
            insights['potential_issues'].append("Low mood variety - system may be stuck in patterns")
            insights['recommendations'].append("Consider adjusting mood detection sensitivity or forcing evolution")
        
        # Analyze semantic analysis quality
        if self.semantic_analyses:
            recent_analyses = list(self.semantic_analyses)[-20:]
            avg_confidence = sum(a['analysis']['confidence'] for a in recent_analyses) / len(recent_analyses)
            
            if avg_confidence < 0.5:
                insights['potential_issues'].append(f"Low semantic analysis confidence: {avg_confidence:.2f}")
                insights['recommendations'].append("Check NLP model availability and input quality")
        
        # Analyze performance
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-50:]
            avg_duration = sum(m['duration_ms'] for m in recent_metrics) / len(recent_metrics)
            
            if avg_duration > 1500:
                insights['potential_issues'].append(f"High average processing time: {avg_duration:.0f}ms")
                insights['recommendations'].append("Consider performance optimization or caching improvements")
        
        # Check warning rates
        total_warnings = sum(self.warning_counts.values())
        if total_warnings > 10:
            insights['potential_issues'].append(f"High warning count: {total_warnings}")
            insights['recommendations'].append("Review system logs for recurring issues")
        
        # Positive observations
        if mood_analysis.get('mood_health') == 'healthy':
            insights['observations'].append("Mood system showing healthy variety and transitions")
        
        if not insights['potential_issues']:
            insights['observations'].append("No significant issues detected - system operating normally")
        
        return insights


# Global monitor instance for easy access
_global_monitor = None

def get_language_monitor() -> LanguageSystemMonitor:
    """Get global language system monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = LanguageSystemMonitor()
    return _global_monitor