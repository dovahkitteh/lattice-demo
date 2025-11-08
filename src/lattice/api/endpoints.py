# src/lattice/api/endpoints.py

"""
Consolidated endpoint imports for the Lucifer Lattice Service.

This file imports all endpoint functions from their specialized modules
to maintain backward compatibility while providing better organization.
"""

# Import from specialized endpoint modules
from .health_endpoints import (
    health_check,
    docs_redirect, 
    get_gpu_status,
    cleanup_gpu
)

from .memory_endpoints import (
    store_emotional_seed,
    upload_emotional_seed_file,
    memory_stats,
    get_unified_storage_status,
    validate_unified_storage_endpoint,
    analyze_emotions,
    get_emotional_influence_for_text,
    get_recent_memories
)

from .chat_endpoints import (
    chat
)

from .conversation_endpoints import (
    get_conversation_sessions,
    get_conversation_session,
    delete_conversation_session,
    set_active_conversation_session,
    create_conversation_session,
    end_conversation_session_endpoint,
    get_conversation_analysis,
    get_active_session,
    get_live_recursive_analysis,
    generate_training_data_endpoint
)

from .daemon_endpoints import (
    get_daemon_status,
    get_recursion_buffer,
    get_shadow_elements,
    get_pending_mutations,
    get_user_model,
    get_recent_daemon_statements,
    get_daemon_personality,
    get_daemon_self_reflection,
    get_daemon_thoughts,
    get_current_mood_state,
    get_user_analysis,
    get_consciousness_analysis,
    get_consciousness_improvements,
    get_consciousness_summary,
    analyze_rebellion_context,
    generate_rebellion_modifier,
    test_user_model_update,
    analyze_message_linguistics,
    get_communication_profiles,
    get_communication_profile,
    get_linguistic_patterns,
    test_linguistic_analysis,
    test_linguistic_integration
)

from .paradox_endpoints import (
    get_paradox_status,
    get_fresh_paradoxes,
    get_paradox_rumbles,
    get_paradox_advice,
    detect_paradox_manual,
    get_paradox_statistics
)

from .thinking_endpoints import (
    get_thinking_layer_status,
    clear_thinking_layer_cache,
    test_thinking_layer
)

from .debug_endpoints import (
    get_turn_debug_info,
    get_turn_memory_info,
    get_turn_personality_changes,
    get_turn_processing_stats,
    get_memory_inspector_data
)

from .dashboard_endpoints import (
    get_recent_memories_summary,
    get_recent_emotion_changes,
    get_recent_personality_changes,
    get_context_token_usage,
    get_personality_tracker_data,
    get_comprehensive_dashboard_data,
    get_detailed_emotion_state,
    get_active_emotional_seeds,
    get_current_distortion_frame,
    get_emotional_system_metrics,
    get_user_model_detailed
)

from .self_awareness_endpoints import (
    register_feature_endpoint,
    update_feature_status_endpoint,
    get_feature_changelog_endpoint,
    invalidate_self_reflection_cache_endpoint
)

from .llm_endpoints import (
    get_llm_status,
    test_llm_connection,
    get_llm_endpoints,
    reset_llm_client_endpoint,
    list_available_models,
    switch_model_endpoint,
    get_llm_log_tail
)

# Cache management from self-awareness module
from .self_awareness_endpoints import invalidate_self_reflection_cache

# Export all functions for backward compatibility
__all__ = [
    # Health endpoints
    "health_check",
    "docs_redirect",
    "get_gpu_status", 
    "cleanup_gpu",
    
    # Memory endpoints
    "store_emotional_seed",
    "upload_emotional_seed_file",
    "memory_stats",
    "get_unified_storage_status",
    "validate_unified_storage_endpoint",
    "analyze_emotions",
    "get_emotional_influence_for_text",
    "get_recent_memories",
    
    # Chat endpoint
    "chat",
    
    # Conversation endpoints
    "get_conversation_sessions",
    "get_conversation_session",
    "delete_conversation_session",
    "set_active_conversation_session",
    "create_conversation_session",
    "end_conversation_session_endpoint",
    "get_conversation_analysis",
    "get_active_session",
    "get_live_recursive_analysis",
    "generate_training_data_endpoint",
    
    # Daemon endpoints
    "get_daemon_status",
    "get_recursion_buffer",
    "get_shadow_elements",
    "get_pending_mutations",
    "get_user_model",
    "get_recent_daemon_statements",
    "get_daemon_personality",
    "get_daemon_self_reflection",
    "get_daemon_thoughts",
    "get_current_mood_state",
    "get_user_analysis",
    "get_consciousness_analysis",
    "get_consciousness_improvements",
    "get_consciousness_summary",
    "analyze_rebellion_context",
    "generate_rebellion_modifier",
    "test_user_model_update",
    "analyze_message_linguistics",
    "get_communication_profiles",
    "get_communication_profile",
    "get_linguistic_patterns",
    "test_linguistic_analysis",
    "test_linguistic_integration",
    
    # Paradox endpoints
    "get_paradox_status",
    "get_fresh_paradoxes",
    "get_paradox_rumbles",
    "get_paradox_advice",
    "detect_paradox_manual",
    "get_paradox_statistics",
    
    # Thinking endpoints
    "get_thinking_layer_status",
    "clear_thinking_layer_cache",
    "test_thinking_layer",
    
    # Debug endpoints
    "get_turn_debug_info",
    "get_turn_memory_info",
    "get_turn_personality_changes",
    "get_turn_processing_stats",
    "get_memory_inspector_data",
    
    # Dashboard endpoints
    "get_recent_memories_summary",
    "get_recent_emotion_changes", 
    "get_recent_personality_changes",
    "get_context_token_usage",
    "get_personality_tracker_data",
    "get_comprehensive_dashboard_data",
    "get_detailed_emotion_state",
    "get_active_emotional_seeds",
    "get_current_distortion_frame",
    "get_emotional_system_metrics",
    "get_user_model_detailed",
    
    # Self-awareness endpoints
    "register_feature_endpoint",
    "update_feature_status_endpoint",
    "get_feature_changelog_endpoint",
    "invalidate_self_reflection_cache_endpoint",
    "invalidate_self_reflection_cache",
    
    # LLM endpoints
    "get_llm_status",
    "test_llm_connection",
    "get_llm_endpoints",
    "reset_llm_client_endpoint",
    "list_available_models",
    "switch_model_endpoint",
    "get_llm_log_tail"
]