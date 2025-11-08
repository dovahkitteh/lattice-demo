"""
User Modeling Module

Handles deep user personality modeling through emotional analysis
and post-conversation processing.
"""

from .post_conversation_analyzer import (
    PostConversationAnalyzer,
    ConversationInsight,
    UserModelAnalysis,
    post_conversation_analyzer
)

from .unified_user_model import (
    UnifiedUserModel,
    PersonalityComponent,
    UnifiedUserModelManager,
    unified_user_model_manager
)

from .chat_integration import (
    UserModelingChatIntegration,
    user_modeling_chat_integration
)

__all__ = [
    "PostConversationAnalyzer",
    "ConversationInsight", 
    "UserModelAnalysis",
    "post_conversation_analyzer",
    "UnifiedUserModel",
    "PersonalityComponent",
    "UnifiedUserModelManager",
    "unified_user_model_manager",
    "UserModelingChatIntegration",
    "user_modeling_chat_integration"
]
