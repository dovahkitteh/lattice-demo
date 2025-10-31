# src/lattice/emotions/seeds.py
"""
Manages the loading, retrieval, and scheduling of emotional seeds.
This module is responsible for:
- Loading the seed catalog from a configuration file.
- Implementing the retrieval algorithm to select appropriate seeds based on the current emotional state.
- Managing the homeostatic drive to schedule counter-seeds for emotional regulation.
"""
import logging
import numpy as np
from typing import List, Tuple

from ..models import EmotionState, Seed, ScheduledCounterSeed, UserModel, AppraisalBuffer
from ..config import get_emotion_config

logger = logging.getLogger(__name__)

# This will be our placeholder for tracking how long a dimension is out of bounds
# DEPRECATED: This is now handled by the homeostatic_counters in the EmotionState model.
# HOMEOSTATIC_COUNTERS = {
#     "attachment_security_low": 0,
#     "narrative_fusion_high": 0,
# }

def schedule_counter_seeds(current_state: EmotionState) -> List[ScheduledCounterSeed]:
    """
    Checks homeostatic setpoints and schedules counter-seeds if thresholds are breached.
    This function now reads/writes counters from the state object itself, making it stateful.
    """
    scheduled_list = []
    config = get_emotion_config().config
    setpoints = config.get("setpoints", {})
    counter_seed_rules = config.get("counter_seeds", [])
    
    # --- Attachment Security Drive ---
    attachment_setpoint = setpoints.get("attachment_security", [0.6, 0.95])  # Updated to match new config
    low_attachment_key = "attachment_security_low"
    if current_state.attachment_security < attachment_setpoint[0]:
        current_state.homeostatic_counters[low_attachment_key] = current_state.homeostatic_counters.get(low_attachment_key, 0) + 1
    else:
        current_state.homeostatic_counters[low_attachment_key] = 0 # Reset counter

    # --- Narrative Fusion Drive ---
    narrative_setpoint = setpoints.get("narrative_fusion", [0.3, 0.9])
    high_fusion_key = "narrative_fusion_high"
    if current_state.narrative_fusion > narrative_setpoint[1]:
        current_state.homeostatic_counters[high_fusion_key] = current_state.homeostatic_counters.get(high_fusion_key, 0) + 1
    else:
        current_state.homeostatic_counters[high_fusion_key] = 0 # Reset counter

    # Check rules against the state's counters
    for rule in counter_seed_rules:
        condition_key = rule.get("condition_key") # e.g., "attachment_security_low"
        trigger_threshold = rule.get("trigger_after_turns", 3) # Get threshold from config
        seed_id = rule.get("seed_id")
        
        if not all([condition_key, trigger_threshold, seed_id]):
            logger.warning(f"Skipping invalid counter-seed rule: {rule}")
            continue

        current_counter = current_state.homeostatic_counters.get(condition_key, 0)

        if current_counter >= trigger_threshold:
            # Check if this seed is already scheduled to avoid duplicates
            if not any(s.seed_id == seed_id for s in scheduled_list):
                from .metrics import metrics_manager # Import locally to avoid circular dependency
                reason = f"{condition_key} exceeded threshold ({current_counter} >= {trigger_threshold})"
                scheduled_list.append(ScheduledCounterSeed(seed_id=seed_id, trigger_turn=0, reason=reason))
                logger.info(f"Homeostatic drive triggered: Scheduling '{seed_id}' because {reason}.")
                # Record the latency for this regulation event
                metrics_manager.record_regulation_latency(current_counter)
            
    return scheduled_list

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def _evaluate_activation_condition(condition: str, current_state: EmotionState, appraisal: AppraisalBuffer = None) -> bool:
    """
    Evaluates a single activation condition against the current state.
    
    Supported condition formats:
    - "unconditional" -> always True
    - "valence:gt:0.7" -> current_state.valence > 0.7
    - "valence:lt:0.3" -> current_state.valence < 0.3
    - "attachment_security:lt:0.5" -> current_state.attachment_security < 0.5
    - "self_cohesion:gt:0.8" -> current_state.self_cohesion > 0.8
    - "creative_expansion:gt:0.6" -> current_state.creative_expansion > 0.6
    - "instability_index:gt:0.5" -> current_state.instability_index > 0.5
    - "narrative_fusion:gt:0.9" -> current_state.narrative_fusion > 0.9
    - "trigger_type:attention_withdrawal" -> check if trigger type is present in appraisal
    - "contrast_event:valence_mismatch" -> check if contrast event type is present in appraisal
    - "homeostatic_drive:attachment_security" -> check if homeostatic counter is active
    """
    if condition == "unconditional":
        return True
    
    # Parse condition format: "dimension:operator:value" or "trigger_type:type_name"
    parts = condition.split(":")
    if len(parts) < 2:
        logger.warning(f"Invalid activation condition format: {condition}")
        return False
    
    if parts[0] == "trigger_type":
        if not appraisal:
            return False
        trigger_type = ":".join(parts[1:])  # Handle trigger types with colons
        return any(trigger['type'].lower() == trigger_type.lower() for trigger in appraisal.triggers)
    
    elif parts[0] == "contrast_event":
        if not appraisal:
            return False
        event_type = ":".join(parts[1:])
        return any(event['type'] == event_type for event in appraisal.contrast_events)
    
    elif parts[0] == "homeostatic_drive":
        drive_name = parts[1]
        counter_key = f"{drive_name}_low" if drive_name == "attachment_security" else f"{drive_name}_high"
        return current_state.homeostatic_counters.get(counter_key, 0) > 0
    
    elif len(parts) == 3:
        # Format: "dimension:operator:value"
        dimension, operator, value_str = parts
        try:
            value = float(value_str)
        except ValueError:
            logger.warning(f"Invalid numeric value in condition: {condition}")
            return False
        
        # Get current dimension value
        dimension_value = getattr(current_state, dimension, None)
        if dimension_value is None:
            logger.warning(f"Unknown dimension in condition: {dimension}")
            return False
        
        # Apply operator
        if operator == "gt":
            return dimension_value > value
        elif operator == "lt":
            return dimension_value < value
        elif operator == "eq":
            return abs(dimension_value - value) < 0.001  # Float comparison
        elif operator == "gte":
            return dimension_value >= value
        elif operator == "lte":
            return dimension_value <= value
        else:
            logger.warning(f"Unknown operator in condition: {operator}")
            return False
    
    logger.warning(f"Unrecognized activation condition format: {condition}")
    return False


def retrieve_relevant_seeds(
    current_state: EmotionState, 
    scheduled_seeds: List[ScheduledCounterSeed],
    appraisal: AppraisalBuffer = None
) -> List[Seed]:
    """
    Retrieves the most relevant emotional seeds based on the current state.

    Args:
        current_state: The agent's current EmotionState.
        scheduled_seeds: A list of any counter-seeds scheduled by homeostatic drives.

    Returns:
        A list of the most relevant Seed objects.
    """
    emotion_config_manager = get_emotion_config()
    all_seeds = emotion_config_manager.seeds
    config = emotion_config_manager.config
    
    if not all_seeds:
        logger.warning("Seed catalog is empty. No seeds will be retrieved.")
        return []

    # Find the config for the current mood family
    mood_family_config = {}
    for family in config.get("families", []):
        if family.get("name") == current_state.mood_family:
            mood_family_config = family
            break

    # Determine retrieval breadth based on config flag OR state heuristics (state wins)
    config_flag = mood_family_config.get("expansive_flag")
    heuristic_flag = False
    try:
        heuristic_flag = (
            (current_state.valence is not None and float(current_state.valence) >= 0.6)
            or (getattr(current_state, "creative_expansion", 0.0) >= 0.6)
        ) and (current_state.attachment_security is None or float(current_state.attachment_security) >= 0.6)
    except Exception:
        heuristic_flag = False

    # Heuristic should be able to override conservative config defaults during runtime
    is_expansive = bool(config_flag) or heuristic_flag
    
    if is_expansive:
        k = config.get("retrieval", {}).get("expansive_seed_k", 7)
        logger.debug(f"Operating in EXPANSIVE retrieval mode for mood '{current_state.mood_family}'. K={k}")
    else:
        k = config.get("retrieval", {}).get("narrow_seed_k", 4)
        logger.debug(f"Operating in NARROWING retrieval mode for mood '{current_state.mood_family}'. K={k}")

    # Score all available seeds
    scored_seeds: List[Tuple[float, Seed]] = []
    fallback_scored: List[Tuple[float, Seed]] = []  # candidates that don't meet activation
    mood_bonuses = config.get("mood_family_congruence", {})
    unconditional_seeds: List[Seed] = []

    for seed in all_seeds:
        # Check activation conditions
        is_activated = False
        if not seed.activation_conditions:
            # No conditions means always active
            is_activated = True
        else:
            # Check if any activation condition is met (OR logic)
            for condition in seed.activation_conditions:
                if _evaluate_activation_condition(condition, current_state, appraisal):
                    is_activated = True
                    break
        
        # Special handling for unconditional seeds
        if "unconditional" in seed.activation_conditions:
            unconditional_seeds.append(seed)
            continue
        
        similarity = _cosine_similarity(current_state.vector_28, seed.self_affect_vector)
        
        # Add mood_family_congruence_bonus
        mood_bonus = mood_bonuses.get(current_state.mood_family, 1.0)
        
        if not is_activated:
            # Keep lower-weighted fallback so retrieval breadth can still reach k
            fallback_scored.append((similarity * seed.retrieval_importance * mood_bonus * 0.75, seed))
            continue

        score = similarity * seed.retrieval_importance * mood_bonus
        scored_seeds.append((score, seed))

    # Sort seeds by score in descending order
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    fallback_scored.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize the final list with any scheduled counter-seeds and unconditional seeds
    final_seeds: List[Seed] = []
    final_seeds.extend(unconditional_seeds)
    
    scheduled_seed_ids = {s.seed_id for s in scheduled_seeds}
    if scheduled_seed_ids:
        logger.debug(f"Force-including {len(scheduled_seed_ids)} scheduled counter-seeds: {scheduled_seed_ids}")
        for seed in all_seeds:
            if seed.id in scheduled_seed_ids:
                final_seeds.append(seed)

    # Add the top-k retrieved seeds, ensuring we respect the limit 'k' and avoid duplicates
    retrieved_seed_ids = {s.id for s in final_seeds}
    
    # We only count the dynamically retrieved seeds towards our limit `k`
    dynamically_retrieved_count = 0
    for score, seed in scored_seeds:
        if dynamically_retrieved_count >= k:
            break
        
        if seed.id not in retrieved_seed_ids:
            final_seeds.append(seed)
            retrieved_seed_ids.add(seed.id)
            dynamically_retrieved_count += 1

    # If still short of k, backfill with best fallback candidates
    for score, seed in fallback_scored:
        if dynamically_retrieved_count >= k:
            break
        if seed.id in retrieved_seed_ids:
            continue
        final_seeds.append(seed)
        retrieved_seed_ids.add(seed.id)
        dynamically_retrieved_count += 1

    if logger.getEffectiveLevel() == logging.DEBUG:
        log_output = [f"'{s.id}' (Score: {score:.4f})" for score, s in scored_seeds[:10]]
        logger.debug(f"Top 10 candidate seeds: {', '.join(log_output)}")
        logger.debug(f"Final selected seeds: {[s.id for s in final_seeds]}")
        
    # Ensure we return exactly k items to satisfy callers expecting a fixed-size set
    return final_seeds[:k]

async def retrieve_and_integrate_seeds(
    current_state: EmotionState,
    appraisal: AppraisalBuffer,
    user_model: UserModel
) -> Tuple[EmotionState, List[Seed]]:
    """
    Full pipeline for seed retrieval and integration.
    """
    # 1. Check for and schedule counter-seeds based on homeostatic drives
    scheduled_seeds = schedule_counter_seeds(current_state)

    # 2. Retrieve relevant seeds, including scheduled ones
    active_seeds = retrieve_relevant_seeds(current_state, scheduled_seeds, appraisal)
    
    if not active_seeds and not appraisal.spike_adjustments:
        # No seeds and no spikes, no state change needed from this module
        return current_state, []

    # 3. Integrate Seeds and Spikes into a new state object
    new_state = current_state.model_copy(deep=True)
    
    # Apply spike adjustments first (immediate reaction)
    for spike in appraisal.spike_adjustments:
        for i, val in enumerate(spike):
            new_state.vector_28[i] += val

    # Then apply seed influences (deeper, narrative-driven changes)
    for seed in active_seeds:
        for i in range(len(new_state.vector_28)):
            new_state.vector_28[i] += seed.self_affect_vector[i] * seed.personality_influence

    # Normalize the vector to prevent runaway values, keeping them in a [0, 1] range.
    max_val = max(new_state.vector_28) if any(new_state.vector_28) else 0
    if max_val > 1.0:
        new_state.vector_28 = [v / max_val for v in new_state.vector_28]
        
    # Clamp values to be non-negative
    new_state.vector_28 = [max(0.0, v) for v in new_state.vector_28]

    logger.debug(f"Integrated {len(active_seeds)} seeds and {len(appraisal.spike_adjustments)} spikes.")

    return new_state, active_seeds 