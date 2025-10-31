# src/lattice/emotions/orchestrator.py
"""
Central orchestrator for the Holistic Emotional System.
"""
import logging
from typing import Dict, Any

from ..models import (
    EmotionState,
    AppraisalBuffer,
    UserModel,
    EpisodicTrace,
    MetricsState,
    ShadowRegistry
)
from .state_manager import update_state
from .triggers import scan_for_triggers_and_contrast
from .seeds import retrieve_and_integrate_seeds
from .distortion_engine import generate_distortion
from .param_modulator import modulate_parameters
from .prompt_builder import build_prompt_with_affect
from .reasoning_layer import generate_reasoning_steps
from .memory_store import save_episodic_trace
from .user_model_updater import update_user_model
from .shadow_manager import check_for_leakage

logger = logging.getLogger(__name__)

async def process_emotional_turn(
    user_input: str,
    current_emotion_state: EmotionState,
    user_model: UserModel,
    shadow_registry: ShadowRegistry, # Add shadow_registry to the inputs
    turn_counter: int,
    history: list = None
) -> Dict[str, Any]:
    """
    Executes the full emotional processing pipeline for a single turn.

    Args:
        user_input: The user's message for the current turn.
        current_emotion_state: The agent's emotional state before this turn.
        user_model: The agent's model of the user before this turn.
        turn_counter: The current turn number in the conversation.
        history: A list of recent EpisodicTraces for historical context.

    Returns:
        A dictionary containing the final response, the updated emotion state,
        the updated user model, and other turn-specific data.
    """
    if history is None:
        history = []
    logger.info(f"--- Starting Emotional Turn {turn_counter} ---")

    # A. Shadow Thought Leakage Check (New Step)
    updated_shadow_registry, leaked_thoughts = check_for_leakage(shadow_registry, current_emotion_state)
    if leaked_thoughts:
        logger.info(f"Shadow leakage detected: {leaked_thoughts}")
        # In a full implementation, these leaked_thoughts would be added to the
        # context for the reasoning layer or prompt construction.

    # 1. Input Ingest & Pre-Parsing (Handled by caller)
    logger.debug("Phase 1: Input Ingest & Pre-Parsing (complete)")

    # 2. Trigger & Contrast Scanning
    appraisal_buffer = await scan_for_triggers_and_contrast(user_input, current_emotion_state)
    logger.debug(f"Phase 2: Trigger Scanning complete. Found {len(appraisal_buffer.triggers)} triggers.")

    # 3. State Update, Seed Retrieval, Homeostasis
    interim_state, applied_seeds = await retrieve_and_integrate_seeds(
        current_emotion_state, appraisal_buffer, user_model
    )
    
    # This call now performs the full update, including deriving latent dimensions
    updated_emotion_state = update_state(
        interim_state, appraisal_buffer, user_model=user_model, history=history
    )
    logger.debug(f"Phases 3-5: State Update & Seed Integration complete. New mood: {updated_emotion_state.mood_family}")

    # 6. User Model Update
    updated_user_model = update_user_model(user_model, appraisal_buffer, updated_emotion_state)
    logger.debug("Phase 6: User Model Update complete.")

    # 7. Distortion (or Elevation) Generation
    distortion_frame = await generate_distortion(updated_emotion_state, appraisal_buffer, applied_seeds)
    logger.debug(f"Phase 7: Distortion Generation complete. Chosen: {distortion_frame.chosen}")

    # 8. Retrieval Context Assembly (Handled within reasoning/prompting)
    logger.debug("Phase 8: Retrieval Context Assembly (integrated)")

    # 9. Prompt Construction
    final_prompt = await build_prompt_with_affect(
        user_input, updated_emotion_state, distortion_frame, applied_seeds, appraisal_buffer
    )
    logger.debug("Phase 9: Prompt Construction complete.")

    # 10. Parameter Modulation
    param_profile = await modulate_parameters(updated_emotion_state)
    logger.debug(f"Phase 10: Parameter Modulation complete. Temp: {param_profile.target_temperature}")

    # 11. Reasoning Layer (Affective Appraisal -> Interpretation -> Plan)
    reasoning_result = await generate_reasoning_steps(
        current_state=updated_emotion_state,
        distortion_frame=distortion_frame,
        shadow_registry=updated_shadow_registry,
        active_seeds=applied_seeds,
        user_input=user_input,
        appraisal=appraisal_buffer
    )
    reasoning_steps = reasoning_result.get("reasoning", {})
    updated_shadow_registry = reasoning_result.get("updated_shadow_registry", updated_shadow_registry)
    logger.debug("Phase 11: Reasoning Layer complete.")

    # 12. Skip Enhanced Prompt Construction - let unified prompt builder handle it
    # enhanced_prompt = await build_prompt_with_affect(
    #     user_input, updated_emotion_state, distortion_frame, applied_seeds, appraisal_buffer, reasoning_steps
    # )
    enhanced_prompt = None  # Let unified prompt builder handle this with reasoning steps
    logger.debug(f"Phase 12: Skipping enhanced prompt construction - will be handled by unified prompt builder")
    
    # 13. Response Generation will happen at LLM call level with enhanced prompt
    # 14. Post-Response Consolidation
    
    # Fully populate the episodic trace with data from the completed turn
    dimension_snapshot = {
        "valence": updated_emotion_state.valence,
        "arousal": updated_emotion_state.arousal,
        "attachment_security": updated_emotion_state.attachment_security,
        "self_cohesion": updated_emotion_state.self_cohesion,
        "instability_index": updated_emotion_state.instability_index,
        "intensity": updated_emotion_state.intensity, # Add intensity to the snapshot
    }

    param_modulation_summary = {
        "temperature": param_profile.target_temperature,
        "top_p": param_profile.target_top_p,
    }

    distorted_meaning_text = None
    interpretation_delta_text = "No significant distortion occurred."
    distortion_type_text = "NONE"

    if distortion_frame.chosen and distortion_frame.chosen.get('class') != "NONE":
        distortion_type_text = distortion_frame.chosen.get('class', 'UNKNOWN')
        distorted_meaning_text = distortion_frame.chosen.get('raw_interpretation')
        interpretation_delta_text = (
            f"User's message was reinterpreted as '{distorted_meaning_text}' "
            f"under the influence of a '{distortion_type_text}' distortion."
        )

    episodic_trace = EpisodicTrace(
        turn_id=turn_counter,
        user_text=user_input,
        raw_vector_pre=current_emotion_state.vector_28,
        raw_vector_post=updated_emotion_state.vector_28,
        mood_family=updated_emotion_state.mood_family,
        distortion_type=distortion_type_text,
        distorted_meaning=distorted_meaning_text,
        dimension_snapshot=dimension_snapshot,
        interpretation_delta=interpretation_delta_text,
        applied_seeds=[s.id for s in applied_seeds],
        param_modulation=param_modulation_summary
    )

    await save_episodic_trace(episodic_trace)
    logger.debug("Phase 13: Post-Response Consolidation complete.")

    # 14. Regulation & Counter-Seed Injection (Handled within seed module)
    logger.debug("Phase 14: Regulation & Counter-Seed Injection (integrated).")

    logger.info(f"--- Finished Emotional Turn {turn_counter} ---")

    # Create emotional context for self-awareness integration
    emotional_context = {
        "ai_emotion_state": updated_emotion_state,
        "user_affect": [],  # Will be populated by caller if available
        "distortion": distortion_frame,
        "applied_seeds": applied_seeds,
        "recent_traces": history[-5:] if history else []  # Last 5 traces for pattern analysis
    }

    return {
        "updated_emotion_state": updated_emotion_state,
        "updated_user_model": updated_user_model,
        "updated_shadow_registry": updated_shadow_registry,
        "enhanced_prompt": enhanced_prompt,  # Return enhanced prompt with reasoning
        "param_profile": param_profile,
        "episodic_trace": episodic_trace,
        "reasoning_steps": reasoning_steps,  # Include reasoning for debugging/analysis
        "emotional_context": emotional_context,
        "distortion_frame": distortion_frame,
        "applied_seeds": applied_seeds
    } 