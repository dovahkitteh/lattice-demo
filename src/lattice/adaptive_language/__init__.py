"""
ADAPTIVE LANGUAGE SYSTEM
Semantic-driven conversational adaptation for consciousness evolution

This module replaces the monolithic language_hygiene.py with a modular,
NLP-powered architecture that enables natural, organic fluctuations in
AI responses across a wide spectrum of conversational states.
"""

from .core.orchestrator import AdaptiveLanguageOrchestrator
from .core.models import (
    ConversationContext,
    SemanticAnalysis,
    MoodState,
    LanguageStyle
)
from .legacy.compatibility import (
    # Legacy compatibility functions
    build_adaptive_mythic_prompt,
    build_mythic_prompt,
    clean_clinical_language,
    daemon_responds,
    architect_says,
    get_mood_state,
    reset_mood_system
)
from .debug import (
    LanguageSystemMonitor,
    ConversationAnalyzer,
    PerformanceProfiler
)

# Main interface - backward compatible
adaptive_language_system = AdaptiveLanguageOrchestrator()

__all__ = [
    'AdaptiveLanguageOrchestrator',
    'ConversationContext',
    'SemanticAnalysis', 
    'MoodState',
    'LanguageStyle',
    'adaptive_language_system',
    'build_adaptive_mythic_prompt',
    'build_mythic_prompt',
    'clean_clinical_language',
    'daemon_responds',
    'architect_says',
    'get_mood_state',
    'reset_mood_system',
    'LanguageSystemMonitor',
    'ConversationAnalyzer',
    'PerformanceProfiler'
]