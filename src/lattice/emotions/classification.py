import logging
from typing import List

from ..config import DEVICE, GOEMO_LABEL2IDX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMOTION CLASSIFICATION FUNCTIONS
# ---------------------------------------------------------------------------

async def classify_affect(text: str) -> list[float]:
    """Legacy single-channel affect classification (kept for backward compatibility)"""
    from ..config import classifier
    if classifier is None:
        logger.error("Classifier not initialized")
        return [0.0] * 28
    scores = classifier(text)[0]  # list of dicts w/ label + score
    vec = [0.0] * 28  # Updated for 28 emotions including neutral
    for s in scores:
        if s["label"] in GOEMO_LABEL2IDX:
            idx = GOEMO_LABEL2IDX[s["label"]]
            vec[idx] = s["score"]
    return vec

async def classify_user_affect(text: str) -> list[float]:
    """Classify user emotional state from their message"""
    try:
        from ..config import classifier
        if classifier is None:
            logger.error("Classifier not initialized")
            return [0.0] * 28
        scores = classifier(text)[0]  # list of dicts w/ label + score
        vec = [0.0] * 28  # 28 emotions including neutral
        for s in scores:
            if s["label"] in GOEMO_LABEL2IDX:
                idx = GOEMO_LABEL2IDX[s["label"]]
                vec[idx] = s["score"]
        return vec
    except Exception as e:
        logger.error(f"Error classifying user affect: {e}")
        return [0.0] * 28  # neutral fallback

async def classify_llm_affect(text: str) -> list[float]:
    """Classify LLM's emotional state from its response"""
    try:
        from ..config import classifier
        if classifier is None:
            logger.error("Classifier not initialized")
            return [0.0] * 28
        # For now, use the same classifier but we could add LLM-specific prompting
        # or fine-tuning for self-reflection emotions in the future
        scores = classifier(text)[0]  # list of dicts w/ label + score
        vec = [0.0] * 28  # 28 emotions including neutral
        for s in scores:
            if s["label"] in GOEMO_LABEL2IDX:
                idx = GOEMO_LABEL2IDX[s["label"]]
                vec[idx] = s["score"]
        return vec
    except Exception as e:
        logger.error(f"Error classifying LLM affect: {e}")
        return [0.0] * 28  # neutral fallback

def get_top_emotions(affect_vector: List[float], top_k: int = 3) -> List[str]:
    """Get the top K emotions from an affect vector"""
    if not affect_vector or len(affect_vector) != 28:
        return ["neutral"]
    
    # Get indices of top emotions
    emotion_scores = list(enumerate(affect_vector))
    emotion_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_emotions = []
    for i in range(min(top_k, len(emotion_scores))):
        idx, score = emotion_scores[i]
        if score > 0.1:  # Threshold for significance
            # Find emotion name by index
            emotion_name = [k for k, v in GOEMO_LABEL2IDX.items() if v == idx][0]
            top_emotions.append(emotion_name)
    
    return top_emotions if top_emotions else ["neutral"]

def get_emotion_summary(affect_vector: List[float]) -> dict:
    """Get a comprehensive summary of emotions in the affect vector"""
    if not affect_vector or len(affect_vector) != 28:
        return {"error": "Invalid affect vector"}
    
    # Calculate statistics
    total_magnitude = sum(abs(x) for x in affect_vector)
    max_emotion_idx = affect_vector.index(max(affect_vector))
    max_emotion = [k for k, v in GOEMO_LABEL2IDX.items() if v == max_emotion_idx][0]
    
    # Get emotions above threshold
    significant_emotions = []
    for i, score in enumerate(affect_vector):
        if score > 0.1:
            emotion_name = [k for k, v in GOEMO_LABEL2IDX.items() if v == i][0]
            significant_emotions.append({
                "emotion": emotion_name,
                "score": score
            })
    
    # Sort by score
    significant_emotions.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "total_magnitude": total_magnitude,
        "dominant_emotion": max_emotion,
        "dominant_score": affect_vector[max_emotion_idx],
        "significant_emotions": significant_emotions[:5],  # Top 5
        "emotion_count": len(significant_emotions),
        "top_emotions": [e["emotion"] for e in significant_emotions[:3]]
    } 