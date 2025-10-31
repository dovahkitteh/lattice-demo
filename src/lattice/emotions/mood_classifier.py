# src/lattice/emotions/mood_classifier.py
"""
Assigns a mood family to the current emotional state.
This module will:
- Use a table-driven approach based on the mood family rules defined in the configuration.
- Determine the dominant mood family for a given emotional state vector.
- Handle priority and override logic to select a single mood family per turn.
"""
import logging
from typing import Dict, Any, List

from ..models import EmotionState
from ..config import get_emotion_config

logger = logging.getLogger(__name__)

def _check_condition(state_value: float, condition: Dict[str, float]) -> bool:
    """Checks if a state value meets a simple condition (e.g., {'gt': 0.5})."""
    if "gt" in condition and not state_value > condition["gt"]:
        return False
    if "lt" in condition and not state_value < condition["lt"]:
        return False
    if "gte" in condition and not state_value >= condition["gte"]:
        return False
    if "lte" in condition and not state_value <= condition["lte"]:
        return False
    if "eq" in condition and not state_value == condition["eq"]:
        return False
    return True

def classify_mood(current_state: EmotionState) -> str:
    """
    Classifies the current emotional state into a mood family.

    Args:
        current_state: The agent's current EmotionState.

    Returns:
        The name of the assigned mood family.
    """
    emotion_config_manager = get_emotion_config()
    families = emotion_config_manager.config.get("families", [])
    
    # Sort families by priority (highest first)
    sorted_families = sorted(families, key=lambda x: x.get("priority", 0), reverse=True)
    
    logger.debug(f"Starting mood classification for state (Valence: {current_state.valence:.2f}, Arousal: {current_state.arousal:.2f})")

    for family in sorted_families:
        name = family.get("name", "Unknown Family")
        criteria = family.get("criteria", {})
        
        conditions_met = True
        
        # Check all defined criteria for the family
        for dimension, condition in criteria.items():
            state_value = getattr(current_state, dimension, None)
            
            if state_value is None:
                logger.warning(f"Dimension '{dimension}' not found in EmotionState for mood '{name}'.")
                conditions_met = False
                break
            
            if not _check_condition(state_value, condition):
                logger.debug(f"Mood '{name}' rejected: {dimension} ({state_value:.2f}) failed condition {condition}.")
                conditions_met = False
                break
        
        if conditions_met:
            logger.info(f"Mood classified as: '{name}'")
            return name
            
    # Fallback case: If no specific criteria are met, default to Serene Attunement.
    # This is the designated lowest-priority, general-purpose mood.
    fallback_mood = "Serene Attunement"
    logger.warning(f"No specific mood family criteria met. Falling back to default: '{fallback_mood}'")
    return fallback_mood 