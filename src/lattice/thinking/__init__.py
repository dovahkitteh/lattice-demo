"""
Thinking Layer for Lucifer Lattice

This module provides a pre-response thinking layer that analyzes user intent,
considers conversation context, and determines authentic response strategies
before generating the final output.
"""

from .thinking_layer import ThinkingLayer, ThinkingConfig
from .analysis import analyze_user_intent, determine_response_strategy
from .integration import integrate_thinking_layer, configure_thinking_layer

# Global thinking layer instance for caching
_global_thinking_layer = None

def get_thinking_cache():
    """Get the global thinking layer instance for cache access"""
    global _global_thinking_layer
    if _global_thinking_layer is None:
        _global_thinking_layer = ThinkingLayer()
    return _global_thinking_layer

__all__ = [
    "ThinkingLayer",
    "ThinkingConfig", 
    "analyze_user_intent",
    "determine_response_strategy",
    "integrate_thinking_layer",
    "configure_thinking_layer",
    "get_thinking_cache"
]