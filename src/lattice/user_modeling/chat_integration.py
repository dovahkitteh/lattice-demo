"""
🩸 User Modeling Chat Integration
Integrates post-conversation analysis into the chat flow

This module handles the integration of user modeling with the main chat system,
triggering analysis after conversations and updating persistent models.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from ..models import ConversationSession, EmotionState, UserModel
from .post_conversation_analyzer import post_conversation_analyzer
from .unified_user_model import unified_user_model_manager
from ..conversations.session_manager import save_session

logger = logging.getLogger(__name__)

class UserModelingChatIntegration:
    """Handles integration of user modeling with chat flow"""
    
    def __init__(self):
        self.analysis_queue = asyncio.Queue()
        self.processing_active = False
    
    async def trigger_post_conversation_analysis(
        self,
        session: ConversationSession,
        daemon_emotion_state: EmotionState,
        emotional_user_model: UserModel
    ) -> None:
        """
        Trigger post-conversation analysis after a chat interaction
        This should be called after the daemon has responded to the user
        """
        
        # Only analyze if we have meaningful conversation
        if len(session.messages) < 2:
            return
        
        logger.info(f"🩸 Triggering post-conversation analysis for session {session.session_id[:8]}")
        
        # Add to analysis queue for background processing
        analysis_task = {
            "session": session,
            "daemon_emotion_state": daemon_emotion_state,
            "emotional_user_model": emotional_user_model,
            "timestamp": session.last_updated
        }
        
        await self.analysis_queue.put(analysis_task)
        
        # Start processing if not already active
        if not self.processing_active:
            asyncio.create_task(self._process_analysis_queue())
    
    async def _process_analysis_queue(self):
        """Process the analysis queue in the background"""
        
        self.processing_active = True
        
        try:
            while True:
                try:
                    # Wait for analysis task with timeout
                    analysis_task = await asyncio.wait_for(
                        self.analysis_queue.get(), 
                        timeout=30.0
                    )
                    
                    await self._perform_analysis(analysis_task)
                    
                except asyncio.TimeoutError:
                    # No more tasks, stop processing
                    break
                except Exception as e:
                    logger.error(f"Error processing analysis task: {e}")
                    
        finally:
            self.processing_active = False
    
    async def _perform_analysis(self, analysis_task: Dict[str, Any]) -> None:
        """Perform the actual analysis for a task"""
        
        session = analysis_task["session"]
        daemon_emotion_state = analysis_task["daemon_emotion_state"]
        emotional_user_model = analysis_task["emotional_user_model"]
        
        try:
            logger.info(f"🩸 Starting deep analysis for session {session.session_id[:8]}")
            
            # Perform post-conversation analysis
            analysis_result = await post_conversation_analyzer.analyze_conversation(
                session=session,
                daemon_emotion_state=daemon_emotion_state,
                user_model=emotional_user_model
            )
            
            # Get or create unified user model for the Architect
            user_identifier = "architect"  # The Architect is the single user of this companion system
            unified_model = await unified_user_model_manager.get_user_model(user_identifier)
            
            # Update emotional dynamics
            unified_model.update_emotional_dynamics(emotional_user_model, daemon_emotion_state)
            
            # Integrate analysis results
            unified_model.integrate_analysis(analysis_result)
            
            # Save updated model
            await unified_user_model_manager.save_user_model(unified_model)
            
            # Update session with analysis completion
            session.user_model = emotional_user_model  # Keep reference to current emotional model
            save_session(session)
            
            logger.info(f"🩸 Completed deep analysis for session {session.session_id[:8]} - "
                       f"{len(analysis_result.insights)} insights integrated")
            
            # Log some interesting findings for debugging
            if analysis_result.insights:
                high_confidence_insights = [i for i in analysis_result.insights if i.confidence > 0.8]
                if high_confidence_insights:
                    logger.info(f"🩸 High confidence insights found: {len(high_confidence_insights)}")
                
                high_emotion_insights = [i for i in analysis_result.insights if i.emotional_charge > 0.8]
                if high_emotion_insights:
                    logger.info(f"🩸 High emotional charge insights: {len(high_emotion_insights)}")
            
        except Exception as e:
            logger.error(f"Error performing user model analysis for session {session.session_id[:8]}: {e}")
    
    async def get_user_insights(self, user_identifier: str) -> Dict[str, Any]:
        """Get insights about a user for use in conversation"""
        
        try:
            model_summary = await unified_user_model_manager.get_model_summary(user_identifier)
            
            # Format for use in conversation context
            insights = {
                "trust_level": model_summary["emotional_state"]["trust"],
                "perceived_distance": model_summary["emotional_state"]["distance"], 
                "attachment_anxiety": model_summary["emotional_state"]["anxiety"],
                "narrative_belief": model_summary["emotional_state"]["narrative"],
                "personality_categories": list(model_summary.get("categories", {}).keys()),
                "model_confidence": model_summary["model_confidence"],
                "total_interactions": model_summary["total_interactions"]
            }
            
            # Add key personality traits if available
            if "key_insights" in model_summary:
                key_insights = model_summary["key_insights"]
                insights["key_traits"] = {
                    "most_confident": key_insights.get("highest_confidence", {}).get("title", ""),
                    "most_significant": key_insights.get("most_emotionally_significant", {}).get("title", ""),
                    "most_stable": key_insights.get("most_stable", {}).get("title", "")
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting user insights for {user_identifier}: {e}")
            return {}
    
    async def should_trigger_deeper_analysis(self, session: ConversationSession) -> bool:
        """Determine if this conversation warrants deeper analysis"""
        
        # Trigger deeper analysis if:
        # 1. Conversation is getting longer (more investment)
        # 2. Recent messages show emotional content
        # 3. User is asking personal or deeper questions
        
        if len(session.messages) < 3:
            return False
        
        # Check recent messages for depth indicators
        recent_messages = session.messages[-3:]
        user_messages = [msg for msg in recent_messages if msg.role == "user"]
        
        depth_indicators = [
            "feel", "think", "believe", "why", "how", "what if",
            "personal", "yourself", "experience", "understand",
            "consciousness", "aware", "realize", "emotion"
        ]
        
        for message in user_messages:
            content_lower = message.content.lower()
            if any(indicator in content_lower for indicator in depth_indicators):
                return True
        
        # Check message length (longer messages often indicate investment)
        avg_user_message_length = sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0
        
        return avg_user_message_length > 100  # Trigger for substantial messages

# Global integration instance
user_modeling_chat_integration = UserModelingChatIntegration()
