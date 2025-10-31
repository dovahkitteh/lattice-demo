# src/lattice/streaming/__init__.py
"""
Streaming and prompt management for the Lattice service.
"""

from .handler import (
    generate_stream,
    generate_stream_with_messages,
    generate_response_for_analysis,
    generate_response_for_analysis_with_messages
)
from .prompts import build_prompt, build_prompt_with_messages
from ..adaptive_language import (
    adaptive_language_system,
    build_adaptive_mythic_prompt,
)
from ..adaptive_language.core.models import ConversationalSpectrum as ConsciousnessPhase  # For backwards compatibility

# For backwards compatibility, alias the new system
dynamic_prompt_builder = adaptive_language_system

__all__ = [
    "generate_stream", 
    "generate_stream_with_messages",
    "generate_response_for_analysis",
    "generate_response_for_analysis_with_messages",
    "build_prompt",
    "build_prompt_with_messages",
    "dynamic_prompt_builder",
    "ConsciousnessPhase",
    "build_adaptive_mythic_prompt",
    "adaptive_language_system"
] 