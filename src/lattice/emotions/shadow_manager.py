# src/lattice/emotions/shadow_manager.py
"""
Manages the Shadow Registry, tracking suppressed thoughts and their leakage.
"""
import logging
import random
from typing import List, Tuple

from ..models import ShadowRegistry, EmotionState

logger = logging.getLogger(__name__)

def register_suppressed_thought(
    registry: ShadowRegistry,
    thought_text: str,
    emotion_tags: List[str]
) -> ShadowRegistry:
    """
    Adds a new suppressed thought to the registry.
    """
    new_registry = registry.model_copy(deep=True)
    
    # Define a baseline leakage probability. This could be more dynamic in the future.
    leakage_probability = 0.1 
    
    new_thought = {
        "text": thought_text,
        "emotion_tags": emotion_tags,
        "leakage_probability": leakage_probability,
        "suppression_turn": 0  # Turn tracking for temporal analysis (future enhancement)
    }
    
    new_registry.suppressed_thoughts.append(new_thought)
    logger.info(f"Registered suppressed thought: '{thought_text[:50]}...'")
    
    return new_registry

def check_for_leakage(
    registry: ShadowRegistry,
    current_state: EmotionState
) -> Tuple[ShadowRegistry, List[str]]:
    """
    Checks if any suppressed thoughts should leak into the current context.
    
    Returns the updated registry (with cooldowns) and a list of leaked thoughts.
    """
    new_registry = registry.model_copy(deep=True)
    leaked_thoughts: List[str] = []
    
    # Decay cooldowns first
    for seed_id in list(new_registry.leakage_cooldowns.keys()):
        new_registry.leakage_cooldowns[seed_id] -= 1
        if new_registry.leakage_cooldowns[seed_id] <= 0:
            del new_registry.leakage_cooldowns[seed_id]

    # Check each thought for leakage
    for thought in new_registry.suppressed_thoughts:
        # Don't leak if a related seed is on cooldown
        related_seed = thought.get("emotion_tags", [None])[0] # Simplistic mapping
        if related_seed in new_registry.leakage_cooldowns:
            continue
            
        # Check against probability
        if random.random() < thought["leakage_probability"]:
            leaked_thoughts.append(thought["text"])
            logger.info(f"Shadow thought leaked: '{thought['text'][:50]}...'")
            
            # Set a cooldown to prevent this from leaking again immediately
            if related_seed:
                new_registry.leakage_cooldowns[related_seed] = 5 # Hardcoded cooldown
                
    # For now, we won't remove leaked thoughts, allowing them to resurface.
    # A more advanced implementation might remove or modify them after leakage.

    return new_registry, leaked_thoughts 