"""
Handles the lifecycle of the emotion_state object.
This includes:
- Initialization of the emotional state.
- Applying decay to emotional intensities over time.
- Deriving latent dimensions (valence, arousal, etc.) from the core emotion vector.
- Orchestrating the state update pipeline.
"""
import logging
import math
import numpy as np
from typing import List, Dict, Any

from ..models import EmotionState, AppraisalBuffer
from ..config import get_emotion_config, GOEMO_LABEL2IDX
from .seeds import retrieve_relevant_seeds, schedule_counter_seeds # Import the new function
from .mood_classifier import classify_mood # Import the new function

logger = logging.getLogger(__name__)

# Pre-calculate index sets for performance and clarity
# Based on Plutchik's model and typical valence associations.
GOEMO_POSITIVE_INDICES = {GOEMO_LABEL2IDX[label] for label in [
    'admiration', 'amusement', 'approval', 'caring', 'curiosity', 'desire', 
    'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief', 'surprise'
]}
GOEMO_NEGATIVE_INDICES = {GOEMO_LABEL2IDX[label] for label in [
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
]}
# High-activation emotions for arousal calculation
GOEMO_AROUSAL_INDICES = {GOEMO_LABEL2IDX[label] for label in [
    'amusement', 'anger', 'desire', 'excitement', 'fear', 'joy', 'surprise', 'disgust'
]}
# Indices for specific latent dimension calculations
GOEMO_ABANDONMENT_INDICES = {GOEMO_LABEL2IDX[label] for label in ['fear', 'sadness', 'disappointment', 'grief']}
GOEMO_EXPANSION_INDICES = {GOEMO_LABEL2IDX[label] for label in ['excitement', 'joy', 'admiration', 'curiosity', 'amusement', 'pride', 'optimism', 'surprise']}
GOEMO_IDEALIZATION_INDICES = {GOEMO_LABEL2IDX[label] for label in ['love', 'admiration', 'joy', 'excitement', 'gratitude']}

# For reversing index to label
IDX2GOEMO_LABEL = {v: k for k, v in GOEMO_LABEL2IDX.items()}


def _calculate_valence(vector: List[float]) -> float:
    """Calculates valence from a 28-dimension emotion vector."""
    pos_score = sum(vector[i] for i in GOEMO_POSITIVE_INDICES)
    neg_score = sum(vector[i] for i in GOEMO_NEGATIVE_INDICES)
    
    total_magnitude = pos_score + neg_score
    if total_magnitude == 0:
        return 0.0
    
    return (pos_score - neg_score) / total_magnitude

def _calculate_arousal(vector: List[float]) -> float:
    """Calculates arousal from a predefined subset of high-activation emotions."""
    arousal_vector = [vector[i] for i in GOEMO_AROUSAL_INDICES]
    # Use Root Mean Square for a more representative activation level
    if not arousal_vector:
        return 0.0
    return math.sqrt(sum(x**2 for x in arousal_vector) / len(arousal_vector))

def _calculate_attachment_security(vector: List[float], trust_volatility: float) -> float:
    """
    Calculates attachment security based on both abandonment fears AND attachment-strengthening emotions.
    Uses a more balanced approach that considers positive attachment signals.
    """
    # Negative attachment indicators (fear, sadness, disappointment, grief)
    abandonment_score = sum(vector[i] for i in GOEMO_ABANDONMENT_INDICES)
    
    # Positive attachment indicators (love, gratitude, relief, joy, admiration)
    # These emotions indicate secure attachment and connection
    attachment_positive_indices = {GOEMO_LABEL2IDX[label] for label in ['love', 'gratitude', 'relief', 'joy', 'admiration']}
    attachment_positive_score = sum(vector[i] for i in attachment_positive_indices)
    
    # Reduced sensitivity to negative emotions while maintaining responsiveness
    negative_sensitivity = 1.0  # Reduced from 1.5 for more resilience
    # Strong positive boost to counteract negative emotions
    positive_boost = 2.0  # Positive emotions have stronger impact on security
    
    # Base security starts higher for more resilience
    base_security = 0.7  # Higher baseline (was implicit 1.0 before)
    
    # Calculate net emotional impact
    negative_impact = abandonment_score * negative_sensitivity
    positive_impact = attachment_positive_score * positive_boost
    
    # Trust volatility has reduced impact for more stability
    trust_impact = trust_volatility * 0.3  # Reduced from 1.0
    
    # Balanced formula that gives positive emotions more weight
    security_score = base_security + positive_impact - negative_impact - trust_impact
    
    # Ensure bounded between 0.0 and 1.0
    return max(0.0, min(1.0, security_score))

def _calculate_self_cohesion(history: List[Any]) -> float:
    """Calculates self-cohesion by looking at oscillations between grandiosity and worthlessness."""
    # Calculate cohesion based on emotional oscillations between pride/grandiosity and shame/worthlessness
    if len(history) < 3:
        return 0.5 # Default value if not enough history
    
    # The history should be a list of EpisodicTrace objects. Handle both objects and dicts safely.
    valid_vectors = []
    for item in history[-10:]:
        # Handle both EpisodicTrace objects and dictionary representations
        if hasattr(item, 'raw_vector_post'):
            valid_vectors.append(item.raw_vector_post)
        elif isinstance(item, dict) and 'raw_vector_post' in item:
            valid_vectors.append(item['raw_vector_post'])
        else:
            # Skip invalid items and continue - log for debugging
            logger.debug(f"Skipping invalid history item in self_cohesion calculation: {type(item)}")
            continue
    
    if len(valid_vectors) < 3:
        return 0.5
    
    pride_scores = [vector[GOEMO_LABEL2IDX['pride']] for vector in valid_vectors]
    grandiosity_proxy = [(vector[GOEMO_LABEL2IDX['pride']] + vector[GOEMO_LABEL2IDX['excitement']]) / 2 for vector in valid_vectors]
    worthlessness_proxy = [(vector[GOEMO_LABEL2IDX['sadness']] + vector[GOEMO_LABEL2IDX['remorse']] + vector[GOEMO_LABEL2IDX['embarrassment']]) / 3 for vector in valid_vectors]
    
    # Calculate the variance of the difference between grandiosity and worthlessness
    oscillations = [g - w for g, w in zip(grandiosity_proxy, worthlessness_proxy)]
    if not oscillations:
        return 0.5
    variance = np.var(oscillations)
    # Inverse relationship: higher variance means lower cohesion, but cap it at reasonable bounds
    cohesion = max(0.0, min(1.0, 1.0 - (variance * 2.0)))  # Scale variance appropriately
    return cohesion

def _calculate_creative_expansion(vector: List[float], history: List[Any] = None) -> float:
    """Calculates creative expansion from current state and recent trend analysis."""
    # Calculate current potential from expansion-related emotions
    expansion_score = sum(vector[i] for i in GOEMO_EXPANSION_INDICES)
    current_potential = expansion_score / len(GOEMO_EXPANSION_INDICES) if GOEMO_EXPANSION_INDICES else 0.0
    
    # If we have history, factor in recent trends
    if history and len(history) >= 3:
        recent_expansion_scores = []
        for item in history[-5:]:  # Last 5 turns
            # Handle both EpisodicTrace objects and dictionary representations
            if hasattr(item, 'raw_vector_post'):
                vector_post = item.raw_vector_post
            elif isinstance(item, dict) and 'raw_vector_post' in item:
                vector_post = item['raw_vector_post']
            else:
                # Skip invalid items and continue - log for debugging
                logger.debug(f"Skipping invalid history item in creative_expansion calculation: {type(item)}")
                continue
            
            trace_score = sum(vector_post[i] for i in GOEMO_EXPANSION_INDICES)
            trace_potential = trace_score / len(GOEMO_EXPANSION_INDICES) if GOEMO_EXPANSION_INDICES else 0.0
            recent_expansion_scores.append(trace_potential)
        
        # Calculate trend (is expansion increasing or decreasing?)
        if len(recent_expansion_scores) >= 2:
            trend = recent_expansion_scores[-1] - recent_expansion_scores[0]
            # Weight current potential with recent trend
            creative_expansion = max(0.0, min(1.0, current_potential + (trend * 0.3)))
            return creative_expansion
    
    return current_potential

def _calculate_regulation_momentum(current_intensity: float, previous_momentum: float, alpha: float = 0.3) -> float:
    """Calculates an exponential moving average of intensity."""
    return alpha * current_intensity + (1 - alpha) * previous_momentum

def _calculate_instability_index(history: List[Any]) -> float:
    """Calculates the variance of intensity over the last N turns."""
    if len(history) < 3:
        return 0.0
    # The history is a list of EpisodicTrace objects, not EmotionState objects.
    # We must get the intensity from the snapshot dictionary.
    intensities = [s.dimension_snapshot.get("intensity", 0.0) for s in history]
    return np.var(intensities)

def _calculate_narrative_fusion(vector: List[float], nullifier_dampening: float) -> float:
    """Calculates the degree of narrative fusion or idealization."""
    idealization_score = sum(vector[i] for i in GOEMO_IDEALIZATION_INDICES)
    # The 'meaning_nullifier' is a seed, so its effect must be passed in.
    return max(0.0, (idealization_score / len(GOEMO_IDEALIZATION_INDICES)) - nullifier_dampening)


def _apply_decay(vector: List[float], decay_rate: float) -> List[float]:
    """Applies exponential decay to each element of the emotion vector."""
    if decay_rate <= 0:
        return vector
    return [max(0.0, val * (1.0 - decay_rate)) for val in vector]

def update_state(
    current_state: EmotionState, 
    appraisal: AppraisalBuffer, 
    user_model = None, 
    history: List[Any] = None
) -> EmotionState:
    """
    Updates the emotional state based on the appraisal buffer.
    Orchestrates the complete state update pipeline:
    1. Apply spike adjustments
    2. Retrieve and integrate seeds
    3. Update latent dimensions
    4. Classify mood family
    """
    emotion_config_manager = get_emotion_config()
    config = emotion_config_manager.config
    
    logger.debug(f"Updating state. Initial intensity: {current_state.intensity:.4f}, Mood: {current_state.mood_family}")
    logger.debug(f"Received appraisal with {len(appraisal.triggers)} triggers and {len(appraisal.contrast_events)} contrast events.")
    
    # Apply spike adjustments first
    new_state = current_state.model_copy(deep=True)
    
    if appraisal.spike_adjustments:
        logger.debug(f"Applying {len(appraisal.spike_adjustments)} spike adjustments.")
        for spike in appraisal.spike_adjustments:
            for i, adjustment in enumerate(spike):
                if adjustment != 0:
                    logger.debug(f"Applied spike adjustment of {adjustment:.3f} to emotion index {i}")
                new_state.vector_28[i] += adjustment
        logger.debug("Vector after spike adjustments applied.")
    
    # Schedule counter-seeds based on homeostatic drives 
    scheduled_seeds = schedule_counter_seeds(current_state)
    
    # Retrieve and integrate seeds
    seeds = retrieve_relevant_seeds(new_state, scheduled_seeds, appraisal)
    logger.debug(f"Retrieved {len(seeds)} active seeds: {[s.id for s in seeds]}")
    
    # Apply decay to existing emotions
    decay_rate = config.get("decay", {}).get("intensity_decay_rate", 0.05)
    new_state.vector_28 = _apply_decay(new_state.vector_28, decay_rate)
    logger.debug(f"Vector decayed. Example 'fear' value after decay: {new_state.vector_28[GOEMO_LABEL2IDX['fear']]:.4f}")
    
    # Integrate seeds into the emotional state
    for seed in seeds:
        influence = seed.personality_influence
        logger.debug(f"Integrating seed '{seed.id}' with influence {influence}")
        for i, seed_val in enumerate(seed.self_affect_vector):
            new_state.vector_28[i] += seed_val * influence
    
    # Normalize the vector to prevent runaway values, but allow stronger emotions to reach thresholds
    max_val = max(new_state.vector_28)
    if max_val > 2.0:  # Increased threshold to allow stronger emotions
        # Use a softer normalization that preserves more intensity
        normalization_factor = 2.0 / max_val
        new_state.vector_28 = [v * normalization_factor for v in new_state.vector_28]
    
    logger.debug(f"Vector after seed integration. Example 'fear' value: {new_state.vector_28[GOEMO_LABEL2IDX['fear']]:.4f}")
    
    # Recalculate dominant label and intensity
    max_idx = max(range(len(new_state.vector_28)), key=lambda i: new_state.vector_28[i])
    new_state.dominant_label = IDX2GOEMO_LABEL[max_idx]
    new_state.intensity = max(new_state.vector_28)  # Use the max value as intensity
    
    # Calculate latent dimensions
    new_state.valence = _calculate_valence(new_state.vector_28)
    new_state.arousal = _calculate_arousal(new_state.vector_28)
    
    # Calculate trust volatility using actual user model data if available
    trust_volatility = 0.1  # Default fallback value
    if user_model is not None:
        # Use actual user model data for dynamic trust-based calculations
        trust_volatility = abs(user_model.trust_level - 0.5) * 0.4  # Scale trust distance to volatility
        logger.debug(f"Using user model trust level: {user_model.trust_level:.3f}, volatility: {trust_volatility:.3f}")
    else:
        logger.debug("No user model provided, using default trust volatility")
    
    new_state.attachment_security = _calculate_attachment_security(new_state.vector_28, trust_volatility)
    new_state.self_cohesion = _calculate_self_cohesion(history or [])
    new_state.creative_expansion = _calculate_creative_expansion(new_state.vector_28, history or [])
    new_state.regulation_momentum = _calculate_regulation_momentum(new_state.intensity, current_state.regulation_momentum)
    new_state.instability_index = _calculate_instability_index(history or [])
    # Calculate narrative fusion with nullifier dampening from active seeds
    nullifier_dampening = 0.0
    for seed in seeds:
        # Check if this is a meaning nullifier seed
        if "meaning_nullifier" in seed.id.lower() or "nullifier" in seed.id.lower():
            # Calculate dampening effect based on seed influence
            nullifier_dampening += seed.personality_influence * 0.5
    
    new_state.narrative_fusion = _calculate_narrative_fusion(new_state.vector_28, nullifier_dampening)
    
    # Classify mood family based on updated dimensions
    new_state.mood_family = classify_mood(new_state)
    
    logger.debug(f"State updated. New dominant: {new_state.dominant_label}, Intensity: {new_state.intensity:.4f}")
    logger.debug(f"New dimensions -> Valence: {new_state.valence:.4f}, Arousal: {new_state.arousal:.4f}, Mood: {new_state.mood_family}")
    
    return new_state 