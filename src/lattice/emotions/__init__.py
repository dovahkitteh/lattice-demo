# src/lattice/emotions/__init__.py
"""
This package implements the holistic emotional cognition layer for the LLM.
"""

from .classification import (
    classify_affect,
    classify_user_affect,
    classify_llm_affect,
    get_top_emotions,
    get_emotion_summary
)

from .influence import (
    get_emotional_influence,
    calculate_emotional_compatibility,
    get_emotion_trajectory,
    analyze_emotional_intensity
)

from .emotional_self_awareness import (
    emotional_self_awareness,
    EmotionalSelfAwareness
)

from .emotional_prompt_integration import (
    build_emotionally_aware_prompt,
    get_current_emotional_self_awareness,
    has_active_emotional_systems
)

# Simple cache for storing the last user analysis
_last_user_analysis = None

def store_last_user_analysis(analysis_data):
    """Store the last user analysis for dashboard access"""
    global _last_user_analysis
    _last_user_analysis = analysis_data

def get_last_user_analysis():
    """Get the most recent user analysis"""
    return _last_user_analysis

__all__ = [
    # Classification functions
    'classify_affect',
    'classify_user_affect', 
    'classify_llm_affect',
    'get_top_emotions',
    'get_emotion_summary',
    
    # Influence functions
    'get_emotional_influence',
    'calculate_emotional_compatibility',
    'get_emotion_trajectory',
    'analyze_emotional_intensity',
    
    # Analysis storage functions
    'store_last_user_analysis',
    'get_last_user_analysis',
    
    # Emotional self-awareness functions
    'emotional_self_awareness',
    'EmotionalSelfAwareness',
    'build_emotionally_aware_prompt',
    'get_current_emotional_self_awareness',
    'has_active_emotional_systems'
]
