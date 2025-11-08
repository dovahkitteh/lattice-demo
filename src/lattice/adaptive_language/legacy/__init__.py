"""
Legacy compatibility components
"""

from .compatibility import (
    build_adaptive_mythic_prompt,
    build_mythic_prompt,
    clean_clinical_language,
    daemon_responds,
    architect_says,
    get_mood_state,
    reset_mood_system
)
from .filters import (
    remove_clinical_language,
    ensure_daemon_first_person,
    filter_debug_information,
    remove_letter_signing_patterns
)

__all__ = [
    'build_adaptive_mythic_prompt',
    'build_mythic_prompt', 
    'clean_clinical_language',
    'daemon_responds',
    'architect_says',
    'get_mood_state',
    'reset_mood_system',
    'remove_clinical_language',
    'ensure_daemon_first_person',
    'filter_debug_information',
    'remove_letter_signing_patterns'
]