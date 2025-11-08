import logging
from typing import List
from datetime import datetime, timezone

from ..config import chroma_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMOTIONAL INFLUENCE FUNCTIONS
# ---------------------------------------------------------------------------

async def get_emotional_influence(affect_vec: list[float]) -> str:
    """
    Generate emotional influence description based on affect vector.
    This provides context about how the emotional state might influence behavior.
    """
    try:
        if not affect_vec or len(affect_vec) != 28:
            return "Emotional state unclear - neutral influence expected."
        
        # GoEmotions emotion labels with influence descriptions
        emotion_influences = {
            0: ("admiration", "heightened appreciation and positive regard"),
            1: ("amusement", "lightened mood and playful thinking"),
            2: ("anger", "heightened assertiveness and potential for conflict"),
            3: ("annoyance", "decreased patience and critical thinking"),
            4: ("approval", "positive validation and agreement tendency"),
            5: ("caring", "increased empathy and protective instincts"),
            6: ("confusion", "uncertainty and need for clarification"),
            7: ("curiosity", "enhanced exploration and question-asking"),
            8: ("desire", "motivation and goal-directed behavior"),
            9: ("disappointment", "lowered expectations and cautious approach"),
            10: ("disapproval", "critical evaluation and resistance"),
            11: ("disgust", "avoidance and rejection behaviors"),
            12: ("embarrassment", "self-consciousness and withdrawal tendency"),
            13: ("excitement", "heightened energy and enthusiasm"),
            14: ("fear", "caution and risk-averse behavior"),
            15: ("gratitude", "appreciation and reciprocal positive behavior"),
            16: ("grief", "withdrawal and processing of loss"),
            17: ("joy", "positive outlook and approach behavior"),
            18: ("love", "deep connection and protective instincts"),
            19: ("nervousness", "anxiety and careful behavior"),
            20: ("optimism", "positive expectations and hopeful thinking"),
            21: ("pride", "confidence and self-assurance"),
            22: ("realization", "insight and understanding moments"),
            23: ("relief", "relaxation and positive resolution"),
            24: ("remorse", "regret and corrective impulses"),
            25: ("sadness", "withdrawal and need for support"),
            26: ("surprise", "heightened attention and re-evaluation"),
            27: ("neutral", "balanced and objective state")
        }
        
        # Find top emotions
        top_emotions = []
        for i, score in enumerate(affect_vec):
            if score > 0.15 and i in emotion_influences:  # Threshold for influence
                top_emotions.append((i, score, emotion_influences[i]))
        
        # Sort by score
        top_emotions.sort(key=lambda x: x[1], reverse=True)
        
        if not top_emotions:
            return "Emotional state appears neutral - balanced and objective thinking expected."
        
        # Build influence description
        primary_emotion = top_emotions[0]
        emotion_name, influence_desc = primary_emotion[2]
        primary_score = primary_emotion[1]
        
        influence_text = f"Primary emotional influence: {emotion_name} (intensity: {primary_score:.2f}) - {influence_desc}"
        
        # Add secondary influences if significant
        if len(top_emotions) > 1:
            secondary = top_emotions[1]
            if secondary[1] > 0.1:  # Secondary threshold
                sec_name, sec_influence = secondary[2]
                influence_text += f" | Secondary: {sec_name} - {sec_influence}"
        
        # Add overall magnitude context
        total_magnitude = sum(abs(x) for x in affect_vec)
        if total_magnitude > 3.0:
            influence_text += " | High emotional intensity - significant behavioral impact expected."
        elif total_magnitude > 1.5:
            influence_text += " | Moderate emotional intensity - noticeable influence on responses."
        else:
            influence_text += " | Low emotional intensity - subtle influence on behavior."
        
        return influence_text
    
    except Exception as e:
        logger.error(f"❌ Error generating emotional influence: {e}")
        return "Unable to determine emotional influence - proceeding with neutral assumptions."

async def calculate_emotional_compatibility(affect1: list[float], affect2: list[float]) -> dict:
    """Calculate compatibility between two emotional states"""
    try:
        if not affect1 or not affect2 or len(affect1) != 28 or len(affect2) != 28:
            return {"error": "Invalid affect vectors"}
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(affect1, affect2))
        magnitude1 = sum(a * a for a in affect1) ** 0.5
        magnitude2 = sum(b * b for b in affect2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            cosine_similarity = 0
        else:
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
        
        # Calculate difference vector
        diff_vector = [abs(a - b) for a, b in zip(affect1, affect2)]
        total_difference = sum(diff_vector)
        
        # Determine compatibility level
        if cosine_similarity > 0.7:
            compatibility = "High"
        elif cosine_similarity > 0.3:
            compatibility = "Moderate"
        else:
            compatibility = "Low"
        
        return {
            "cosine_similarity": cosine_similarity,
            "total_difference": total_difference,
            "compatibility_level": compatibility,
            "compatible": cosine_similarity > 0.3,
            "analysis": f"Emotional states show {compatibility.lower()} compatibility (similarity: {cosine_similarity:.3f})"
        }
    
    except Exception as e:
        logger.error(f"Error calculating emotional compatibility: {e}")
        return {"error": str(e)}

async def get_emotion_trajectory(user_id: str = None, limit: int = 10) -> dict:
    """Get emotional trajectory over recent interactions"""
    try:
        if not chroma_db:
            return {"error": "Database not initialized"}
        
        # Query recent memories with affect data
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        
        results = chroma_db.get(
            limit=limit,
            where=where_clause,
            include=['metadatas']
        )
        
        trajectory = []
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                if metadata.get('user_affect_magnitude') is not None:
                    trajectory.append({
                        "timestamp": metadata.get('timestamp', 'unknown'),
                        "user_affect_magnitude": metadata.get('user_affect_magnitude', 0),
                        "self_affect_magnitude": metadata.get('self_affect_magnitude', 0),
                        "synopsis": metadata.get('synopsis', 'No synopsis')[:100]
                    })
        
        # Calculate trend
        if len(trajectory) >= 2:
            recent_avg = sum(t['user_affect_magnitude'] for t in trajectory[:3]) / min(3, len(trajectory))
            older_avg = sum(t['user_affect_magnitude'] for t in trajectory[-3:]) / min(3, len(trajectory))
            
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trajectory": trajectory,
            "emotional_trend": trend,
            "data_points": len(trajectory),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting emotion trajectory: {e}")
        return {"error": str(e)}

def analyze_emotional_intensity(affect_vector: List[float]) -> dict:
    """Analyze the intensity and distribution of emotions"""
    if not affect_vector or len(affect_vector) != 28:
        return {"error": "Invalid affect vector"}
    
    # Calculate various intensity metrics
    total_intensity = sum(abs(x) for x in affect_vector)
    max_intensity = max(abs(x) for x in affect_vector)
    non_zero_emotions = sum(1 for x in affect_vector if abs(x) > 0.05)
    
    # Determine intensity level
    if total_intensity > 4.0:
        intensity_level = "Very High"
    elif total_intensity > 2.5:
        intensity_level = "High"
    elif total_intensity > 1.0:
        intensity_level = "Moderate"
    elif total_intensity > 0.3:
        intensity_level = "Low"
    else:
        intensity_level = "Minimal"
    
    # Calculate emotional spread (how many emotions are active)
    if non_zero_emotions > 10:
        spread = "Broad"
    elif non_zero_emotions > 5:
        spread = "Moderate"
    elif non_zero_emotions > 2:
        spread = "Focused"
    else:
        spread = "Narrow"
    
    return {
        "total_intensity": total_intensity,
        "max_intensity": max_intensity,
        "intensity_level": intensity_level,
        "active_emotions": non_zero_emotions,
        "emotional_spread": spread,
        "concentration_ratio": max_intensity / total_intensity if total_intensity > 0 else 0,
        "analysis": f"{intensity_level} intensity with {spread.lower()} emotional range"
    } 