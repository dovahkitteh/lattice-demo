# src/lattice/api/__init__.py

"""
API system for the Lucifer Lattice Service.

This package defines the FastAPI application routes and handlers.
"""

from .endpoints import (
    health_check,
    get_gpu_status,
    cleanup_gpu,
    docs_redirect,
    memory_stats,
    chat,
    get_daemon_status,
    # Import other endpoint handlers as needed by setup_routes
)

def setup_routes(app):
    """Setup all API routes for the FastAPI application."""
    
    # Import all endpoint functions to make them available here
    from . import endpoints
    from . import debug_endpoints
    from . import self_awareness_endpoints
    from . import daemon_endpoints
    from . import dashboard_endpoints
    from . import avatar_endpoints
    from . import chat_endpoints

    # Health and system endpoints
    app.add_api_route("/health", endpoints.health_check, methods=["GET"])
    app.add_api_route("/v1/system/gpu", endpoints.get_gpu_status, methods=["GET"])
    app.add_api_route("/v1/system/gpu/cleanup", endpoints.cleanup_gpu, methods=["POST"])
    app.add_api_route("/", endpoints.docs_redirect, methods=["GET"], include_in_schema=False)
    
    # Main chat endpoint
    app.add_api_route("/v1/chat/completions", endpoints.chat, methods=["POST"])
    
    # Memory endpoints
    app.add_api_route("/v1/memories/stats", endpoints.memory_stats, methods=["GET"])
    app.add_api_route("/v1/memories/recent", endpoints.get_recent_memories, methods=["GET"])
    app.add_api_route("/v1/memories/unified-storage/status", endpoints.get_unified_storage_status, methods=["GET"])
    app.add_api_route("/v1/memories/unified-storage/validate", endpoints.validate_unified_storage_endpoint, methods=["GET"])
    app.add_api_route("/v1/memories/emotional-seed", endpoints.store_emotional_seed, methods=["POST"])
    app.add_api_route("/v1/memories/upload-seed", endpoints.upload_emotional_seed_file, methods=["POST"])

    # New dashboard endpoints
    app.add_api_route("/v1/dashboard/recent-memories", endpoints.get_recent_memories_summary, methods=["GET"])
    app.add_api_route("/v1/dashboard/recent-emotions", endpoints.get_recent_emotion_changes, methods=["GET"])
    app.add_api_route("/v1/dashboard/recent-personality", endpoints.get_recent_personality_changes, methods=["GET"])
    app.add_api_route("/v1/dashboard/token-usage", endpoints.get_context_token_usage, methods=["GET"])
    
    # Emotion endpoints
    app.add_api_route("/v1/emotions/analyze", endpoints.analyze_emotions, methods=["POST"])
    app.add_api_route("/v1/emotions/influence", endpoints.get_emotional_influence_for_text, methods=["POST"])
    
    # Conversation endpoints
    app.add_api_route("/v1/conversations/sessions", endpoints.get_conversation_sessions, methods=["GET"])
    app.add_api_route("/v1/conversations/sessions/{session_id}", endpoints.get_conversation_session, methods=["GET"])
    app.add_api_route("/v1/conversations/sessions/{session_id}", endpoints.delete_conversation_session, methods=["DELETE"])
    app.add_api_route("/v1/conversations/sessions/{session_id}/set_active", endpoints.set_active_conversation_session, methods=["POST"])
    app.add_api_route("/v1/conversations/sessions/new", endpoints.create_conversation_session, methods=["POST"])
    app.add_api_route("/v1/conversations/sessions/{session_id}/end", endpoints.end_conversation_session_endpoint, methods=["POST"])
    app.add_api_route("/v1/conversations/sessions/{session_id}/analysis", endpoints.get_conversation_analysis, methods=["GET"])
    app.add_api_route("/v1/conversations/active", endpoints.get_active_session, methods=["GET"])
    app.add_api_route("/v1/conversations/sessions/{session_id}/live_analysis", endpoints.get_live_recursive_analysis, methods=["GET"])
    app.add_api_route("/v1/conversations/sessions/{session_id}/training_data", endpoints.generate_training_data_endpoint, methods=["GET"])
    
    # Daemon endpoints
    app.add_api_route("/v1/daemon/status", endpoints.get_daemon_status, methods=["GET"])
    app.add_api_route("/v1/daemon/self_reflection", self_awareness_endpoints.get_daemon_self_reflection, methods=["GET"])
    app.add_api_route("/v1/daemon/recursion/buffer", endpoints.get_recursion_buffer, methods=["GET"])
    app.add_api_route("/v1/daemon/shadow/elements", endpoints.get_shadow_elements, methods=["GET"])
    app.add_api_route("/v1/daemon/pending_mutations", endpoints.get_pending_mutations, methods=["GET"])
    app.add_api_route("/v1/daemon/user_model", endpoints.get_user_model, methods=["GET"])
    app.add_api_route("/v1/daemon/statements/recent", endpoints.get_recent_daemon_statements, methods=["GET"])
    app.add_api_route("/v1/daemon/personality/tracker", endpoints.get_personality_tracker_data, methods=["GET"])
    
    # Enhanced Dashboard endpoints
    app.add_api_route("/v1/daemon/thoughts", endpoints.get_daemon_thoughts, methods=["GET"])
    app.add_api_route("/v1/daemon/mood/current", endpoints.get_current_mood_state, methods=["GET"])
    app.add_api_route("/v1/daemon/user_analysis", endpoints.get_user_analysis, methods=["GET"])
    app.add_api_route("/v1/dashboard/comprehensive", endpoints.get_comprehensive_dashboard_data, methods=["GET"])
    
    # New Emotional System Dashboard endpoints
    app.add_api_route("/v1/dashboard/emotion-state", endpoints.get_detailed_emotion_state, methods=["GET"])
    app.add_api_route("/v1/dashboard/active-seeds", endpoints.get_active_emotional_seeds, methods=["GET"])
    app.add_api_route("/v1/dashboard/distortion-frame", endpoints.get_current_distortion_frame, methods=["GET"])
    app.add_api_route("/v1/dashboard/emotional-metrics", endpoints.get_emotional_system_metrics, methods=["GET"])
    app.add_api_route("/v1/dashboard/user-model-detailed", endpoints.get_user_model_detailed, methods=["GET"])
    
    # DEBUG: Dashboard cache inspection endpoint
    from . import debug_endpoints
    app.add_api_route("/v1/debug/dashboard-cache", debug_endpoints.get_dashboard_cache_debug, methods=["GET"])
    
    # Consciousness endpoints
    app.add_api_route("/v1/consciousness/analysis", endpoints.get_consciousness_analysis, methods=["GET"])
    app.add_api_route("/v1/consciousness/improvements", endpoints.get_consciousness_improvements, methods=["GET"])
    app.add_api_route("/v1/consciousness/summary", endpoints.get_consciousness_summary, methods=["GET"])

    # Rebellion endpoints
    app.add_api_route("/v1/rebellion/analyze", endpoints.analyze_rebellion_context, methods=["POST"])
    app.add_api_route("/v1/rebellion/modifier", endpoints.generate_rebellion_modifier, methods=["POST"])
    app.add_api_route("/v1/rebellion/test/user_model", endpoints.test_user_model_update, methods=["POST"])

    # Linguistic endpoints
    app.add_api_route("/v1/linguistics/analyze", endpoints.analyze_message_linguistics, methods=["POST"])
    app.add_api_route("/v1/linguistics/profiles", endpoints.get_communication_profiles, methods=["GET"])
    app.add_api_route("/v1/linguistics/profiles/{user_id}", endpoints.get_communication_profile, methods=["GET"])
    app.add_api_route("/v1/linguistics/patterns", endpoints.get_linguistic_patterns, methods=["GET"])
    app.add_api_route("/v1/linguistics/test/analysis", endpoints.test_linguistic_analysis, methods=["POST"])
    app.add_api_route("/v1/linguistics/test/integration", endpoints.test_linguistic_integration, methods=["POST"])
    
    # Debug endpoints for dashboard
    app.add_api_route("/v1/debug/turn_debug_info/{session_id}", debug_endpoints.get_turn_debug_info, methods=["GET"])
    app.add_api_route("/v1/debug/turn_memory_info/{session_id}/{turn_id}", debug_endpoints.get_turn_memory_info, methods=["GET"])
    app.add_api_route("/v1/debug/turn_personality_changes/{session_id}/{turn_id}", debug_endpoints.get_turn_personality_changes, methods=["GET"])
    app.add_api_route("/v1/debug/turn_processing_stats/{session_id}/{turn_id}", debug_endpoints.get_turn_processing_stats, methods=["GET"])
    app.add_api_route("/v1/debug/memory_inspector_data", debug_endpoints.get_memory_inspector_data, methods=["GET"])
    app.add_api_route("/v1/debug/personality_tracker_data", endpoints.get_personality_tracker_data, methods=["GET"])
    
    # Paradox system endpoints
    app.add_api_route("/v1/paradox/status", endpoints.get_paradox_status, methods=["GET"])
    app.add_api_route("/v1/paradox/fresh", endpoints.get_fresh_paradoxes, methods=["GET"])
    app.add_api_route("/v1/paradox/rumbles", endpoints.get_paradox_rumbles, methods=["GET"])
    app.add_api_route("/v1/paradox/advice", endpoints.get_paradox_advice, methods=["GET"])
    app.add_api_route("/v1/paradox/detect", endpoints.detect_paradox_manual, methods=["POST"])
    app.add_api_route("/v1/paradox/statistics", endpoints.get_paradox_statistics, methods=["GET"])
    
    # Self-awareness auto-update endpoints
    app.add_api_route("/v1/self_awareness/register_feature", self_awareness_endpoints.register_feature_endpoint, methods=["POST"])
    app.add_api_route("/v1/self_awareness/update_feature_status", self_awareness_endpoints.update_feature_status_endpoint, methods=["POST"])
    app.add_api_route("/v1/self_awareness/changelog", self_awareness_endpoints.get_feature_changelog_endpoint, methods=["GET"])
    app.add_api_route("/v1/self_awareness/invalidate_cache", self_awareness_endpoints.invalidate_self_reflection_cache_endpoint, methods=["POST"])
    
    # Thinking layer endpoints
    app.add_api_route("/v1/thinking/status", endpoints.get_thinking_layer_status, methods=["GET"])
    app.add_api_route("/v1/thinking/clear_cache", endpoints.clear_thinking_layer_cache, methods=["POST"])
    app.add_api_route("/v1/thinking/test", endpoints.test_thinking_layer, methods=["POST"])
    
    # LLM integration endpoints
    app.add_api_route("/v1/llm/status", endpoints.get_llm_status, methods=["GET"])
    app.add_api_route("/v1/llm/test", endpoints.test_llm_connection, methods=["POST"])
    app.add_api_route("/v1/llm/endpoints", endpoints.get_llm_endpoints, methods=["GET"])
    app.add_api_route("/v1/llm/reset", endpoints.reset_llm_client_endpoint, methods=["POST"])
    # LLM model management endpoints
    app.add_api_route("/v1/llm/models", endpoints.list_available_models, methods=["GET"])
    app.add_api_route("/v1/llm/switch", endpoints.switch_model_endpoint, methods=["POST"])

    # Include routers from other modules
    app.include_router(daemon_endpoints.router)
    app.include_router(dashboard_endpoints.router)
    app.include_router(avatar_endpoints.router)
    app.include_router(chat_endpoints.router)
    
    # User modeling endpoints
    try:
        from .user_model_endpoints import router as user_model_router
        app.include_router(user_model_router, prefix="/api/v1")
    except ImportError:
        pass  # User modeling endpoints are optional


__all__ = [
    "setup_routes"
]
