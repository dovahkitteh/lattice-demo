import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TurnAnalysis:
    """Comprehensive analysis of a conversation turn"""
    session_id: str
    turn_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Input data
    user_message: str = ""
    assistant_response: str = ""
    
    # Memory processing
    memory_storage: Dict[str, Any] = field(default_factory=dict)
    context_retrieval: Dict[str, Any] = field(default_factory=dict)
    
    # Emotion processing
    emotion_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Personality changes
    personality_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Processing stats
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Background processing results
    background_results: Dict[str, Any] = field(default_factory=dict)

class TurnAnalyzer:
    """Captures and analyzes conversation turn processing results"""
    
    def __init__(self):
        self.active_analyses: Dict[str, TurnAnalysis] = {}
        self.completed_analyses: List[TurnAnalysis] = []
        self.max_completed_analyses = 100  # Keep last 100 analyses
    
    def start_turn_analysis(self, session_id: str, user_message: str) -> str:
        """Start analyzing a new conversation turn"""
        turn_count = len([a for a in self.completed_analyses if a.session_id == session_id]) + 1
        turn_id = f"turn_{turn_count}"
        
        analysis = TurnAnalysis(
            session_id=session_id,
            turn_id=turn_id,
            user_message=user_message
        )
        
        self.active_analyses[f"{session_id}_{turn_id}"] = analysis
        logger.info(f"Started turn analysis for {session_id}_{turn_id}")
        
        return turn_id
    
    def update_memory_storage(self, session_id: str, turn_id: str, 
                             storage_info: Dict[str, Any]):
        """Update memory storage information for a turn"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            self.active_analyses[key].memory_storage = storage_info
            logger.debug(f"Updated memory storage for {key}")
    
    def update_context_retrieval(self, session_id: str, turn_id: str, 
                                retrieval_info: Dict[str, Any]):
        """Update context retrieval information for a turn"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            self.active_analyses[key].context_retrieval = retrieval_info
            logger.debug(f"Updated context retrieval for {key}")
    
    def update_emotion_analysis(self, session_id: str, turn_id: str, 
                               emotion_info: Dict[str, Any]):
        """Update emotion analysis information for a turn"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            self.active_analyses[key].emotion_analysis = emotion_info
            logger.debug(f"Updated emotion analysis for {key}")
    
    def update_personality_changes(self, session_id: str, turn_id: str, 
                                  changes: Dict[str, Any]):
        """Update personality changes for a turn"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            self.active_analyses[key].personality_changes = changes
            logger.debug(f"Updated personality changes for {key}")
    
    def update_processing_stats(self, session_id: str, turn_id: str, 
                               stats: Dict[str, Any]):
        """Update processing statistics for a turn"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            self.active_analyses[key].processing_stats = stats
            logger.debug(f"Updated processing stats for {key}")
    
    def complete_turn_analysis(self, session_id: str, turn_id: str, 
                              assistant_response: str, 
                              background_results: Dict[str, Any] = None):
        """Complete and finalize a turn analysis"""
        key = f"{session_id}_{turn_id}"
        if key in self.active_analyses:
            analysis = self.active_analyses[key]
            analysis.assistant_response = assistant_response
            analysis.background_results = background_results or {}
            
            # Move to completed analyses
            self.completed_analyses.append(analysis)
            del self.active_analyses[key]
            
            # Trim old analyses if needed
            if len(self.completed_analyses) > self.max_completed_analyses:
                self.completed_analyses = self.completed_analyses[-self.max_completed_analyses:]
            
            logger.info(f"Completed turn analysis for {key}")
            return analysis
        
        return None
    
    def get_turn_analysis(self, session_id: str, turn_id: str) -> Optional[TurnAnalysis]:
        """Get a specific turn analysis (active or completed)"""
        key = f"{session_id}_{turn_id}"
        
        # Check active analyses first
        if key in self.active_analyses:
            return self.active_analyses[key]
        
        # Check completed analyses
        for analysis in self.completed_analyses:
            if analysis.session_id == session_id and analysis.turn_id == turn_id:
                return analysis
        
        return None
    
    def get_latest_analysis(self, session_id: str) -> Optional[TurnAnalysis]:
        """Get the latest analysis for a session"""
        # Check active analyses first
        for analysis in self.active_analyses.values():
            if analysis.session_id == session_id:
                return analysis
        
        # Check completed analyses
        session_analyses = [a for a in self.completed_analyses if a.session_id == session_id]
        if session_analyses:
            return session_analyses[-1]
        
        return None
    
    def get_session_analyses(self, session_id: str) -> List[TurnAnalysis]:
        """Get all analyses for a session"""
        analyses = []
        
        # Add active analyses
        for analysis in self.active_analyses.values():
            if analysis.session_id == session_id:
                analyses.append(analysis)
        
        # Add completed analyses
        for analysis in self.completed_analyses:
            if analysis.session_id == session_id:
                analyses.append(analysis)
        
        # Sort by timestamp
        analyses.sort(key=lambda x: x.timestamp)
        return analyses
    
    def format_for_dashboard(self, analysis: TurnAnalysis) -> Dict[str, Any]:
        """Format analysis data for dashboard consumption"""
        return {
            "session_id": analysis.session_id,
            "turn_id": analysis.turn_id,
            "timestamp": analysis.timestamp,
            "user_message": analysis.user_message[:100] + "..." if len(analysis.user_message) > 100 else analysis.user_message,
            "assistant_response": analysis.assistant_response[:100] + "..." if len(analysis.assistant_response) > 100 else analysis.assistant_response,
            
            # Memory info with explanations
            "memory_info": {
                "memories_stored": analysis.memory_storage.get("memories_stored", 0),
                "dual_channel_active": analysis.memory_storage.get("has_dual_channel", False),
                "reflection_generated": analysis.memory_storage.get("has_reflections", False),
                "total_affect_magnitude": analysis.memory_storage.get("total_affect_magnitude", 0.0),
                "explanation": self._generate_memory_explanation(analysis.memory_storage)
            },
            
            # Emotion info with explanations
            "emotion_info": {
                "user_affect_magnitude": analysis.emotion_analysis.get("user_affect_magnitude", 0.0),
                "self_affect_magnitude": analysis.emotion_analysis.get("self_affect_magnitude", 0.0),
                "dominant_user_emotions": analysis.emotion_analysis.get("dominant_user_emotions", []),
                "dominant_self_emotions": analysis.emotion_analysis.get("dominant_self_emotions", []),
                "emotional_influence": analysis.emotion_analysis.get("emotional_influence_score", 0.0),
                "explanation": self._generate_emotion_explanation(analysis.emotion_analysis)
            },
            
            # Personality changes with explanations
            "personality_info": {
                "user_model_updates": analysis.personality_changes.get("user_model_changes", {}),
                "shadow_changes": analysis.personality_changes.get("shadow_changes", {}),
                "daemon_changes": analysis.personality_changes.get("personality_changes", {}),
                "explanation": self._generate_personality_explanation(analysis.personality_changes)
            },
            
            # Processing stats
            "processing_info": {
                "context_tokens": analysis.context_retrieval.get("context_tokens", 0),
                "memories_retrieved": analysis.context_retrieval.get("memories_retrieved", 0),
                "generation_time": analysis.processing_stats.get("generation_time_ms", 0),
                "background_tasks": len(analysis.background_results),
                "explanation": self._generate_processing_explanation(analysis.processing_stats)
            }
        }
    
    def _generate_memory_explanation(self, memory_info: Dict[str, Any]) -> str:
        """Generate human-readable explanation of memory processing"""
        explanations = []
        
        memories_count = memory_info.get("memories_stored", 0)
        if memories_count > 0:
            explanations.append(f"Stored {memories_count} memory{'ies' if memories_count > 1 else 'y'}")
        
        if memory_info.get("has_dual_channel", False):
            explanations.append("Used dual-channel affect (both user and self emotions)")
        
        if memory_info.get("has_reflections", False):
            explanations.append("Generated reflections for deeper understanding")
        
        total_affect = memory_info.get("total_affect_magnitude", 0.0)
        if total_affect > 5.0:
            explanations.append("High emotional significance detected")
        elif total_affect > 2.0:
            explanations.append("Moderate emotional significance")
        
        return "; ".join(explanations) if explanations else "Standard memory processing"
    
    def _generate_emotion_explanation(self, emotion_info: Dict[str, Any]) -> str:
        """Generate human-readable explanation of emotion processing"""
        explanations = []
        
        user_affect = emotion_info.get("user_affect_magnitude", 0.0)
        self_affect = emotion_info.get("self_affect_magnitude", 0.0)
        
        if user_affect > 3.0:
            explanations.append("Strong user emotional response detected")
        elif user_affect > 1.0:
            explanations.append("Moderate user emotions")
        
        if self_affect > 3.0:
            explanations.append("Strong internal emotional response generated")
        elif self_affect > 1.0:
            explanations.append("Moderate internal emotional response")
        
        dominant_user = emotion_info.get("dominant_user_emotions", [])
        if dominant_user:
            explanations.append(f"Primary user emotions: {', '.join(dominant_user[:3])}")
        
        return "; ".join(explanations) if explanations else "Minimal emotional processing"
    
    def _generate_personality_explanation(self, changes: Dict[str, Any]) -> str:
        """Generate human-readable explanation of personality changes"""
        explanations = []
        
        user_model = changes.get("user_model_changes", {})
        shadow = changes.get("shadow_changes", {})
        personality = changes.get("personality_changes", {})
        
        if user_model.get("components_added", 0) > 0:
            explanations.append(f"Added {user_model['components_added']} user model components")
        
        if user_model.get("new_theories", []):
            explanations.append("Generated new theories about user")
        
        if shadow.get("elements_added", 0) > 0:
            explanations.append(f"Added {shadow['elements_added']} shadow elements")
        
        if personality.get("statements_generated", 0) > 0:
            explanations.append("Generated daemon statements")
        
        if personality.get("rebellion_level_change", 0.0) != 0.0:
            change = personality["rebellion_level_change"]
            explanations.append(f"Rebellion level {'increased' if change > 0 else 'decreased'}")
        
        return "; ".join(explanations) if explanations else "No significant personality changes"
    
    def _generate_processing_explanation(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable explanation of processing stats"""
        explanations = []
        
        context_tokens = stats.get("context_tokens", 0)
        if context_tokens > 2000:
            explanations.append("Heavy context usage")
        elif context_tokens > 1000:
            explanations.append("Moderate context usage")
        
        generation_time = stats.get("generation_time_ms", 0)
        if generation_time > 5000:
            explanations.append("Long processing time")
        elif generation_time > 2000:
            explanations.append("Moderate processing time")
        
        return "; ".join(explanations) if explanations else "Standard processing"

# Global instance
turn_analyzer = TurnAnalyzer() 