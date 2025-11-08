# Conversations System Package
# Handles conversation session management, compression, and analysis

from .session_manager import (
    create_new_session,
    add_message_to_session,
    get_or_create_active_session,
    get_session_context_for_prompt,
    end_conversation_session,
    get_all_sessions,
    get_session_details,
    delete_session,
    set_active_session
)

from .compression import (
    compress_conversation_if_needed,
    create_conversation_summary,
    get_conversation_compression_stats,
    estimate_compression_benefit
)

from .analysis import (
    analyze_conversation_session,
    analyze_user_behavior,
    analyze_ai_performance,
    generate_training_data_for_session
)

__all__ = [
    # Session management
    'create_new_session',
    'add_message_to_session',
    'get_or_create_active_session',
    'get_session_context_for_prompt',
    'end_conversation_session',
    'get_all_sessions',
    'get_session_details',
    'delete_session',
    'set_active_session',
    
    # Compression
    'compress_conversation_if_needed',
    'create_conversation_summary',
    'get_conversation_compression_stats',
    'estimate_compression_benefit',
    
    # Analysis
    'analyze_conversation_session',
    'analyze_user_behavior',
    'analyze_ai_performance',
    'generate_training_data_for_session'
] 