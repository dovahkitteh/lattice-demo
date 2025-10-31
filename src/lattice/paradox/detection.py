"""
PARADOX DETECTION ENGINE
Semantic conflict detection and contradiction analysis
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import re

# Suppress sentence transformers progress bars aggressively
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import transformers
transformers.logging.set_verbosity_error()

# Suppress sentence transformers specific logging
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Patch tqdm for sentence transformers
import tqdm
original_tqdm = tqdm.tqdm

def silent_tqdm(*args, **kwargs):
    kwargs['disable'] = True
    return original_tqdm(*args, **kwargs)

# Apply the patch
tqdm.tqdm = silent_tqdm

from sentence_transformers import SentenceTransformer

# Restore tqdm after import
tqdm.tqdm = original_tqdm

logger = logging.getLogger(__name__)

class ParadoxDetector:
    """Detects semantic conflicts and contradictions in responses"""
    
    def __init__(self, embedder=None, contradiction_threshold: float = 0.7):
        self.embedder = embedder
        self.contradiction_threshold = contradiction_threshold
        self.negation_patterns = [
            r'\b(not|never|no|none|nothing|neither|nowhere|nobody)\b',
            r'\b(don\'?t|doesn\'?t|won\'?t|can\'?t|shouldn\'?t|wouldn\'?t)\b',
            r'\b(impossible|untrue|false|incorrect|wrong)\b'
        ]
        self.contradiction_indicators = [
            r'\b(but|however|although|though|yet|nevertheless|nonetheless)\b',
            r'\b(contradicts?|conflicts?|opposes?)\b',
            r'\b(on the other hand|in contrast|conversely)\b'
        ]
    
    def has_negation_markers(self, text: str) -> bool:
        """Check if text contains negation patterns"""
        text_lower = text.lower()
        for pattern in self.negation_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def has_contradiction_markers(self, text: str) -> bool:
        """Check if text contains contradiction indicators"""
        text_lower = text.lower()
        for pattern in self.contradiction_indicators:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.embedder:
            logger.warning("No embedder available for semantic similarity")
            return 0.0
            
        try:
            embeddings = self.embedder.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def extract_core_claims(self, text: str) -> List[str]:
        """Extract core claims/statements from text"""
        # Simple sentence splitting - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                # Remove common hedging language
                hedging_patterns = [
                    r'\b(I think|I believe|perhaps|maybe|possibly|might|could)\b',
                    r'\b(in my opinion|it seems|appears to be)\b'
                ]
                clean_sentence = sentence
                for pattern in hedging_patterns:
                    clean_sentence = re.sub(pattern, '', clean_sentence, flags=re.IGNORECASE)
                
                if clean_sentence.strip():
                    claims.append(clean_sentence.strip())
        
        return claims
    
    def detect_contradiction_with_belief(self, current_response: str, belief_text: str) -> Tuple[bool, float, str]:
        """
        Detect if current response contradicts a belief
        Returns: (is_contradiction, tension_score, explanation)
        """
        # Extract claims from both texts
        current_claims = self.extract_core_claims(current_response)
        belief_claims = self.extract_core_claims(belief_text)
        
        max_tension = 0.0
        contradiction_found = False
        explanation = ""
        
        for current_claim in current_claims:
            for belief_claim in belief_claims:
                # Check semantic similarity
                similarity = self.semantic_similarity(current_claim, belief_claim)
                
                # If claims are semantically similar but one has negation
                if similarity > 0.6:  # High similarity threshold
                    current_negated = self.has_negation_markers(current_claim)
                    belief_negated = self.has_negation_markers(belief_claim)
                    
                    # Contradiction if one is negated and other isn't
                    if current_negated != belief_negated:
                        tension_score = similarity * 0.9  # High tension for direct contradictions
                        if tension_score > max_tension:
                            max_tension = tension_score
                            contradiction_found = True
                            explanation = f"Direct contradiction: '{current_claim}' vs '{belief_claim}'"
                
                # Check for explicit contradiction markers
                elif (self.has_contradiction_markers(current_claim) and 
                      similarity > 0.4):  # Lower threshold with explicit markers
                    tension_score = similarity * 0.7
                    if tension_score > max_tension:
                        max_tension = tension_score
                        contradiction_found = True
                        explanation = f"Implicit contradiction: '{current_claim}' challenges '{belief_claim}'"
        
        return contradiction_found, max_tension, explanation


async def is_semantic_conflict(current_response: str, belief_base: List[Dict]) -> Tuple[bool, float, Dict]:
    """
    Check if current response conflicts with existing beliefs
    Returns: (has_conflict, max_tension_score, conflict_details)
    """
    from ..config import embedder
    
    detector = ParadoxDetector(embedder=embedder)
    max_tension = 0.0
    conflict_details = {}
    has_conflict = False
    
    for belief in belief_base:
        belief_text = belief.get('text', '') or belief.get('content', '')
        if not belief_text:
            continue
            
        is_contradiction, tension_score, explanation = detector.detect_contradiction_with_belief(
            current_response, belief_text
        )
        
        if is_contradiction and tension_score > max_tension:
            max_tension = tension_score
            has_conflict = True
            conflict_details = {
                'conflicting_belief_id': belief.get('id'),
                'conflicting_belief_text': belief_text,
                'tension_score': tension_score,
                'explanation': explanation,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    # Consider it a conflict if tension exceeds threshold
    has_conflict = has_conflict and max_tension > detector.contradiction_threshold
    
    logger.info(f"Conflict detection: {has_conflict}, max_tension: {max_tension:.3f}")
    if has_conflict:
        logger.info(f"Conflict details: {conflict_details['explanation']}")
    
    return has_conflict, max_tension, conflict_details


async def detect_paradox(current_response: str, context_memories: List[Dict], 
                        affect_delta: float = 0.0) -> Optional[Dict]:
    """
    Comprehensive paradox detection combining semantic conflicts and affective tension
    Returns paradox data if detected, None otherwise
    """
    # Check for semantic conflicts
    has_conflict, tension_score, conflict_details = await is_semantic_conflict(
        current_response, context_memories
    )
    
    if not has_conflict:
        return None
    
    # Calculate combined tension including affective component
    combined_tension = tension_score + (affect_delta * 0.3)  # Weight affective changes
    
    paradox_data = {
        'id': f"paradox_{datetime.now(timezone.utc).timestamp()}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tension_score': combined_tension,
        'semantic_tension': tension_score,
        'affective_delta': affect_delta,
        'status': 'fresh',
        'response_text': current_response,
        'conflict_details': conflict_details,
        'paradox_type': classify_paradox_type(conflict_details.get('explanation', ''))
    }
    
    logger.info(f"Paradox detected: {paradox_data['paradox_type']} (tension: {combined_tension:.3f})")
    
    return paradox_data


def classify_paradox_type(explanation: str) -> str:
    """Classify the type of paradox based on explanation"""
    explanation_lower = explanation.lower()
    
    if 'direct contradiction' in explanation_lower:
        return 'logical_contradiction'
    elif 'implicit contradiction' in explanation_lower:
        return 'semantic_tension'
    elif 'belief' in explanation_lower or 'value' in explanation_lower:
        return 'value_conflict'
    elif 'emotion' in explanation_lower or 'feeling' in explanation_lower:
        return 'affective_dissonance'
    else:
        return 'unknown_tension'