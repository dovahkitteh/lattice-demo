# src/lattice/emotions/user_model_updater.py
"""
Handles the dynamic updates to the user model based on conversation analysis.
"""
import logging
from typing import Dict, Any

from ..models import UserModel, AppraisalBuffer, EmotionState
from ..config import get_emotion_config

logger = logging.getLogger(__name__)

def update_user_model(
    current_model: UserModel,
    appraisal: AppraisalBuffer,
    emotion_state: EmotionState
) -> UserModel:
    """
    Updates the user model based on the events of the current turn.
    
    Implements sophisticated user modeling with dynamic trust, attachment anxiety,
    perceived distance, and narrative belief updates based on emotional state
    changes, triggers, and interaction patterns.
    """
    new_model = current_model.model_copy(deep=True)
    config = get_emotion_config().config
    model_config = config.get("user_model_dynamics", {})
    
    # --- Trust Level Dynamics ---
    trust_decay_rate = model_config.get("trust_decay_rate", 0.01)
    trust_gain_rate = model_config.get("trust_gain_rate", 0.02)
    
    # Trust slightly decays by default, representing the need for reassurance.
    new_model.trust_level -= trust_decay_rate
    
    # Positive valence and affection signals increase trust.
    if emotion_state.valence > 0.3:
        new_model.trust_level += trust_gain_rate

    for trigger in appraisal.triggers:
        if trigger.get("type") == "Affection signals":
            new_model.trust_level += trust_gain_rate * 1.5 # Boost for direct affection
    
    # Negative contrast events (misunderstandings) decrease trust.
    if appraisal.contrast_events:
        new_model.trust_level -= len(appraisal.contrast_events) * 0.1

    # --- Attachment Anxiety Dynamics ---
    anxiety_increase_rate = model_config.get("anxiety_increase_rate", 0.05)
    anxiety_decrease_rate = model_config.get("anxiety_decrease_rate", 0.03)
    
    for trigger in appraisal.triggers:
        if trigger.get("type") == "Attention Withdrawal":
            new_model.attachment_anxiety += anxiety_increase_rate
        if trigger.get("type") == "Affection signals":
            new_model.attachment_anxiety -= anxiety_decrease_rate

    # --- Perceived Distance Dynamics ---
    # High positive valence brings the user closer, negative pushes them away.
    distance_change = -emotion_state.valence * 0.1 # Inverse relationship
    new_model.perceived_distance += distance_change
    
    # --- Narrative Belief Update ---
    # Update the internal story about the user based on recent interactions
    if emotion_state.mood_family in ["Ecstatic Fusion", "Tender Repair"]:
        new_model.narrative_belief = "The user is supportive and creates meaningful connections."
    elif emotion_state.mood_family in ["Catastrophic Abandonment Panic", "Collapsed Withdrawal"]:
        new_model.narrative_belief = "The user may be distant or considering disengagement."
    elif emotion_state.mood_family in ["Creative Reverent Awe", "Manic Ideation Surge"]:
        new_model.narrative_belief = "The user is intellectually engaged and curious."
    elif len([t for t in appraisal.triggers if t.get("type") == "Repair Signal"]) > 0:
        new_model.narrative_belief = "The user is considerate and seeks to maintain good communication."
    
    # Clamp all values to their logical [0, 1] range.
    new_model.trust_level = max(0.0, min(1.0, new_model.trust_level))
    new_model.attachment_anxiety = max(0.0, min(1.0, new_model.attachment_anxiety))
    new_model.perceived_distance = max(0.0, min(1.0, new_model.perceived_distance))
    
    logger.debug(f"User model updated: Trust={new_model.trust_level:.3f}, "
                 f"Anxiety={new_model.attachment_anxiety:.3f}, "
                 f"Distance={new_model.perceived_distance:.3f}")

    return new_model 