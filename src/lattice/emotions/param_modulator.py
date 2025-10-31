# src/lattice/emotions/param_modulator.py
"""
Modulates LLM generation parameters based on the emotional state.
This module will:
- Use a table-driven mapping from mood family to parameter offsets (temperature, top_p, etc.).
- Apply dynamic adjustments based on intensity, valence, and arousal.
- Implement jitter logic for volatile states.
- Apply style-based parameter adjustments for rich emotional expression.
"""
import logging
from typing import Dict, Any

from ..models import EmotionState, ParamProfile
from ..config import get_emotion_config
from .mood_style_modulator import MoodStyleModulator

logger = logging.getLogger(__name__)

# Baseline parameters that offsets will be applied to.
# These can be considered the agent's "default voice".
BASELINE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 3000  # Significantly increased to account for mood-based reductions
}

def _clamp(value, min_val, max_val):
    """Clamps a value within a given range."""
    return max(min_val, min(value, max_val))

async def modulate_parameters(current_state: EmotionState) -> ParamProfile:
    """
    Generates a ParamProfile with modulated LLM parameters based on mood.
    Now includes rich style-based parameter adjustments.
    """
    config = get_emotion_config().config
    mood_families = config.get("families", [])
    
    current_mood_info = next((f for f in mood_families if f["name"] == current_state.mood_family), None)
    
    # Initialize mood style modulator for rich style adjustments
    style_modulator = MoodStyleModulator()
    style_profile = style_modulator.get_style_profile(current_state)
    
    # Start with baseline parameters
    final_params = BASELINE_PARAMS.copy()
    
    if current_mood_info:
        offsets = current_mood_info.get("param_offsets", {})
        logger.debug(f"Applying param offsets for mood '{current_state.mood_family}': {offsets}")
        final_params["temperature"] += offsets.get("temp", 0.0)
        final_params["top_p"] += offsets.get("top_p", 0.0)
        # max_tokens modifier can be a percentage string e.g., "-10%" or direct addition/subtraction
        max_tokens_modifier = offsets.get("max_tokens", 0)
        if isinstance(max_tokens_modifier, str) and max_tokens_modifier.endswith('%'):
            # Handle percentage modification
            percentage = float(max_tokens_modifier[:-1]) / 100
            final_params["max_tokens"] = int(final_params["max_tokens"] * (1 + percentage))
        else:
            # Handle direct addition/subtraction
            final_params["max_tokens"] += max_tokens_modifier
    else:
        logger.warning(f"No param offsets found for mood '{current_state.mood_family}'. Using baseline.")

    # Apply global dynamic adjustments based on latent dimensions
    # As per PARAM_MAPPING.md
    
    # Arousal -> Temperature
    arousal_effect = current_state.arousal * 0.2
    final_params["temperature"] += arousal_effect
    logger.debug(f"Arousal ({current_state.arousal:.2f}) added {arousal_effect:.2f} to temperature.")

    # Valence -> Top_p
    if current_state.valence < -0.8:
        valence_effect = final_params["top_p"] * -0.2 # 20% reduction
        final_params["top_p"] += valence_effect
        logger.debug(f"Extreme low valence ({current_state.valence:.2f}) reduced top_p by {valence_effect:.2f}.")
    elif current_state.valence > 0.8:
        valence_effect = final_params["top_p"] * 0.1 # 10% increase
        final_params["top_p"] += valence_effect
        logger.debug(f"Extreme high valence ({current_state.valence:.2f}) increased top_p by {valence_effect:.2f}.")
        
    # Apply style-based parameter adjustments
    style_adjustments = style_modulator.get_parameter_adjustments(style_profile)
    
    # Apply max tokens multiplier for response length modulation
    if "max_tokens_multiplier" in style_adjustments:
        multiplier = style_adjustments["max_tokens_multiplier"]
        final_params["max_tokens"] = int(final_params["max_tokens"] * multiplier)
        logger.debug(f"Style-based max_tokens adjustment: {multiplier:.2f}x = {final_params['max_tokens']} tokens")
    
    # Apply additional temperature adjustment for cadence
    if "temperature_modifier" in style_adjustments:
        temp_mod = style_adjustments["temperature_modifier"]
        final_params["temperature"] += temp_mod
        logger.debug(f"Style-based temperature adjustment: +{temp_mod:.2f}")
    
    # Clamp values to sane ranges (with expanded range for rich expression)
    final_params["temperature"] = _clamp(final_params["temperature"], 0.1, 2.0)
    final_params["top_p"] = _clamp(final_params["top_p"], 0.1, 1.0)
    final_params["max_tokens"] = _clamp(final_params["max_tokens"], 64, 8192)  # Dramatically expanded range for rich expression
    
    # Implement Jitter logic based on instability_index
    jitter_config = config.get("jitter", {})
    instability_threshold = jitter_config.get("instability_threshold", 0.05)
    jitter_range = [0.0, 0.0]
    jitter_frequency = 0.0
    
    if current_state.instability_index > instability_threshold:
        logger.info(f"Instability index ({current_state.instability_index:.2f}) is high. Applying jitter.")
        if current_state.instability_index > jitter_config.get("high_instability_threshold", 0.1):
            jitter_range = jitter_config.get("high_instability_range", [0.05, 0.15])
        else:
            jitter_range = jitter_config.get("mid_instability_range", [0.01, 0.05])
        jitter_frequency = jitter_config.get("base_frequency", 0.5)


    logger.info(f"Final modulated params: Temp={final_params['temperature']:.2f}, Top_p={final_params['top_p']:.2f}, Max_tokens={final_params['max_tokens']}")
    logger.info(f"Style profile: {style_profile.response_length} length, {style_profile.emotional_intensity} intensity")

    # Create enhanced param profile with style information
    param_profile = ParamProfile(
        target_temperature=final_params["temperature"],
        target_top_p=final_params["top_p"],
        target_max_tokens=final_params["max_tokens"],
        jitter_temperature_range=jitter_range,
        jitter_frequency=jitter_frequency
    )
    
    # Store style profile for use in prompt building
    param_profile.style_profile = style_profile
    
    return param_profile 