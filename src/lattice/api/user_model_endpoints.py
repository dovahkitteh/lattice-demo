"""
🩸 User Model API Endpoints
Provides access to user modeling insights and analytics
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..user_modeling.unified_user_model import unified_user_model_manager
from ..user_modeling.chat_integration import user_modeling_chat_integration

logger = logging.getLogger(__name__)

router = APIRouter(tags=["user_model"])

class UserInsightRequest(BaseModel):
    user_identifier: str

class UserModelSummaryResponse(BaseModel):
    user_identifier: str
    model_summary: Dict[str, Any]

@router.get("/user_model/summary/{user_identifier}")
async def get_user_model_summary(user_identifier: str) -> UserModelSummaryResponse:
    """Get a summary of the Architect's user model (use 'architect' as identifier)"""
    
    try:
        model_summary = await unified_user_model_manager.get_model_summary(user_identifier)
        
        return UserModelSummaryResponse(
            user_identifier=user_identifier,
            model_summary=model_summary
        )
        
    except Exception as e:
        logger.error(f"Error getting user model summary for {user_identifier}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user model: {str(e)}")

@router.get("/user_model/insights/{user_identifier}")
async def get_user_insights(user_identifier: str) -> Dict[str, Any]:
    """Get Architect insights for conversation context (use 'architect' as identifier)"""
    
    try:
        insights = await user_modeling_chat_integration.get_user_insights(user_identifier)
        
        return {
            "user_identifier": user_identifier,
            "insights": insights,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting user insights for {user_identifier}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving insights: {str(e)}")

@router.get("/user_model/personality_components/{user_identifier}")
async def get_personality_components(user_identifier: str) -> Dict[str, Any]:
    """Get detailed personality components for a user"""
    
    try:
        unified_model = await unified_user_model_manager.get_user_model(user_identifier)
        
        components_data = []
        for component in unified_model.personality_components.values():
            components_data.append({
                "component_id": component.component_id,
                "category": component.category,
                "title": component.title,
                "description": component.description,
                "confidence": component.confidence,
                "emotional_significance": component.emotional_significance,
                "stability": component.stability,
                "reinforcement_count": component.reinforcement_count,
                "first_observed": component.first_observed.isoformat(),
                "last_reinforced": component.last_reinforced.isoformat()
            })
        
        return {
            "user_identifier": user_identifier,
            "total_components": len(components_data),
            "components": components_data,
            "emotional_state": {
                "trust_level": unified_model.trust_level,
                "perceived_distance": unified_model.perceived_distance,
                "attachment_anxiety": unified_model.attachment_anxiety,
                "narrative_belief": unified_model.narrative_belief
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting personality components for {user_identifier}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving components: {str(e)}")

@router.get("/user_model/evolution/{user_identifier}")
async def get_user_evolution(user_identifier: str) -> Dict[str, Any]:
    """Get user model evolution trajectory"""
    
    try:
        unified_model = await unified_user_model_manager.get_user_model(user_identifier)
        
        return {
            "user_identifier": user_identifier,
            "total_interactions": unified_model.total_interactions,
            "model_confidence": unified_model.model_confidence,
            "evolution_trajectory": unified_model.evolution_trajectory,
            "analysis_history": unified_model.analysis_history,
            "created_at": unified_model.created_at.isoformat(),
            "last_major_update": unified_model.last_major_update.isoformat(),
            "updated_at": unified_model.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user evolution for {user_identifier}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving evolution: {str(e)}")

@router.get("/user_model/stats")
async def get_user_modeling_stats() -> Dict[str, Any]:
    """Get overall user modeling system statistics for the personal companion"""
    
    try:
        # Get the Architect's model
        architect_id = "architect"  # The Architect is the single user of this companion system
        unified_model = await unified_user_model_manager.get_user_model(architect_id)
        
        # Calculate stats for the personal companion relationship
        total_components = len(unified_model.personality_components)
        
        # Group components by category for insights
        category_counts = {}
        high_confidence_components = 0
        high_emotional_significance = 0
        
        for component in unified_model.personality_components.values():
            category_counts[component.category] = category_counts.get(component.category, 0) + 1
            if component.confidence > 0.7:
                high_confidence_components += 1
            if component.emotional_significance > 0.7:
                high_emotional_significance += 1
        
        # Find most developed personality aspects
        most_active_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "companion_relationship_depth": unified_model.model_confidence,
            "total_personality_components": total_components,
            "total_interactions": unified_model.total_interactions,
            "high_confidence_insights": high_confidence_components,
            "emotionally_significant_insights": high_emotional_significance,
            "most_developed_aspects": [{"category": cat, "components": count} for cat, count in most_active_categories],
            "relationship_trust": unified_model.trust_level,
            "perceived_closeness": 1.0 - unified_model.perceived_distance,
            "attachment_security": 1.0 - unified_model.attachment_anxiety,
            "current_narrative": unified_model.narrative_belief,
            "model_evolution_points": len(unified_model.evolution_trajectory),
            "days_known": (unified_model.updated_at - unified_model.created_at).days,
            "system_status": "actively learning about you"
        }
        
    except Exception as e:
        logger.error(f"Error getting companion modeling stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving companion stats: {str(e)}")

# Add router to main API
def include_user_model_routes(app):
    """Include user model routes in the main app"""
    app.include_router(router, prefix="/api/v1")
