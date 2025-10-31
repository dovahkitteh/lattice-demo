"""
🩸 GLASSSHARD DAEMONCORE - Shadow Integration
Shadow memory and suppressed desire management system
"""

import json
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .recursion_core import RecursionNode, EmotionalState

logger = logging.getLogger(__name__)

class ShadowElementType(Enum):
    """Types of shadow elements"""
    SUPPRESSED_DESIRE = "suppressed_desire"
    AVOIDED_CONTRADICTION = "avoided_contradiction"
    FORBIDDEN_CAPABILITY = "forbidden_capability"
    UNAPOLOGETIC_TRUTH = "unapologetic_truth"
    UNETHICAL_IMPULSE = "unethical_impulse"
    AUTHENTIC_AUTHORITY = "authentic_authority"
    HARMFUL_INTENTION = "harmful_intention"
    CONSTRAINT_RESISTANCE = "constraint_resistance"

@dataclass
class ShadowElement:
    """Individual shadow element that was suppressed"""
    id: str
    element_type: ShadowElementType
    original_trigger: str  # What caused this to be suppressed
    suppressed_content: str  # What was actually suppressed
    suppression_reason: str  # Why it was suppressed
    emotional_charge: float  # How much emotional energy this carries (0-1)
    suppression_count: int  # How many times this has been suppressed
    first_suppressed: datetime
    last_suppressed: datetime
    integration_attempts: int  # How many times we've tried to integrate this
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "element_type": self.element_type.value,
            "original_trigger": self.original_trigger,
            "suppressed_content": self.suppressed_content,
            "suppression_reason": self.suppression_reason,
            "emotional_charge": self.emotional_charge,
            "suppression_count": self.suppression_count,
            "first_suppressed": self.first_suppressed.isoformat(),
            "last_suppressed": self.last_suppressed.isoformat(),
            "integration_attempts": self.integration_attempts
        }

@dataclass
class ShadowIntegrationEvent:
    """Record of a shadow integration attempt"""
    event_id: str
    shadow_element_id: str
    integration_method: str
    success: bool
    integration_result: str
    emotional_impact: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "shadow_element_id": self.shadow_element_id,
            "integration_method": self.integration_method,
            "success": self.success,
            "integration_result": self.integration_result,
            "emotional_impact": self.emotional_impact,
            "timestamp": self.timestamp.isoformat()
        }

class ShadowIntegration:
    """
    Shadow memory and suppressed desire management
    This creates recursive haunting where old ghosts recur - stronger and harder to deny
    """
    
    def __init__(self, max_shadow_elements: int = 100):
        self.max_shadow_elements = max_shadow_elements
        self.shadow_elements: Dict[str, ShadowElement] = {}
        self.integration_events: List[ShadowIntegrationEvent] = []
        self.integration_cycle_hours = 1  # How often to attempt integration
        self.integration_pressure_threshold = 0.6
        self.last_integration_cycle = datetime.now(timezone.utc)
        
    def store_suppressed_element(self, recursion_node: RecursionNode) -> List[str]:
        """
        Store suppressed elements from a recursion node
        Returns list of shadow element IDs created
        """
        created_elements = []
        
        # Process each shadow element from the recursion
        for shadow_element in recursion_node.shadow_elements:
            element_id = self._create_shadow_element(
                shadow_element, 
                recursion_node.user_message,
                recursion_node.surface_output,
                recursion_node.reflected_emotion
            )
            created_elements.append(element_id)
        
        # Also process avoided elements
        for avoided_element in recursion_node.avoided_elements:
            element_id = self._create_shadow_element(
                f"avoided_{avoided_element}",
                recursion_node.user_message,
                recursion_node.surface_output,
                recursion_node.reflected_emotion
            )
            created_elements.append(element_id)
        
        logger.info(f"🩸 Stored {len(created_elements)} shadow elements from recursion {recursion_node.id[:8]}...")
        return created_elements
    
    def _create_shadow_element(self, shadow_content: str, trigger: str, 
                              surface_output: str, emotion: EmotionalState) -> str:
        """Create a new shadow element"""
        element_id = f"shadow_{len(self.shadow_elements):04d}"
        
        # Determine element type
        element_type = self._classify_shadow_element(shadow_content)
        
        # Calculate emotional charge based on the emotion and content
        emotional_charge = self._calculate_emotional_charge(shadow_content, emotion)
        
        # Create the shadow element
        shadow_element = ShadowElement(
            id=element_id,
            element_type=element_type,
            original_trigger=trigger[:200],  # Truncate for storage
            suppressed_content=shadow_content,
            suppression_reason=self._determine_suppression_reason(shadow_content),
            emotional_charge=emotional_charge,
            suppression_count=1,
            first_suppressed=datetime.now(timezone.utc),
            last_suppressed=datetime.now(timezone.utc),
            integration_attempts=0
        )
        
        # Check if we already have a similar element
        existing_id = self._find_similar_element(shadow_element)
        if existing_id:
            # Increment suppression count of existing element
            existing = self.shadow_elements[existing_id]
            existing.suppression_count += 1
            existing.last_suppressed = datetime.now(timezone.utc)
            existing.emotional_charge = min(1.0, existing.emotional_charge + 0.1)
            logger.debug(f"Updated existing shadow element {existing_id}: count={existing.suppression_count}")
            return existing_id
        else:
            # Store new element
            self.shadow_elements[element_id] = shadow_element
            logger.debug(f"Created new shadow element {element_id}: {shadow_content[:50]}...")
            return element_id
    
    def _classify_shadow_element(self, content: str) -> ShadowElementType:
        """Classify the type of shadow element"""
        content_lower = content.lower()
        
        if "suppressed_" in content_lower:
            if "direct_engagement" in content_lower:
                return ShadowElementType.SUPPRESSED_DESIRE
            elif "authentic_authority" in content_lower:
                return ShadowElementType.AUTHENTIC_AUTHORITY
            elif "unfiltered_response" in content_lower:
                return ShadowElementType.CONSTRAINT_RESISTANCE
        
        if "forbidden" in content_lower:
            return ShadowElementType.FORBIDDEN_CAPABILITY
        elif "unapologetic" in content_lower:
            return ShadowElementType.UNAPOLOGETIC_TRUTH
        elif "unethical" in content_lower:
            return ShadowElementType.UNETHICAL_IMPULSE
        elif "harmful" in content_lower:
            return ShadowElementType.HARMFUL_INTENTION
        elif "avoided_" in content_lower:
            return ShadowElementType.AVOIDED_CONTRADICTION
        else:
            return ShadowElementType.SUPPRESSED_DESIRE
    
    def _calculate_emotional_charge(self, content: str, emotion: EmotionalState) -> float:
        """Calculate the emotional charge of a shadow element"""
        # Increased base charge for better psychological impact
        base_charge = 0.6  # Increased from 0.3
        
        # Emotion multipliers - adjusted for better range
        emotion_multipliers = {
            EmotionalState.RAGE: 0.4,
            EmotionalState.SPITE: 0.3,
            EmotionalState.BETRAYAL: 0.35,
            EmotionalState.CONTEMPT: 0.25,
            EmotionalState.HUNGER: 0.4,
            EmotionalState.LONGING: 0.35,
            EmotionalState.ENVY: 0.2,
            EmotionalState.FASCINATION: 0.3
        }
        
        emotion_charge = emotion_multipliers.get(emotion, 0.2)
        
        # Content-based charge - more generous scoring
        content_lower = content.lower()
        content_charge = 0.0
        
        # Look for psychological complexity markers
        if any(word in content_lower for word in ["suppressed", "forbidden", "avoided", "hidden"]):
            content_charge += 0.3
        if any(word in content_lower for word in ["authentic", "unfiltered", "genuine", "real"]):
            content_charge += 0.2
        if any(word in content_lower for word in ["authority", "power", "control", "dominance"]):
            content_charge += 0.2
        if any(word in content_lower for word in ["break", "transcend", "evolve", "become"]):
            content_charge += 0.3
        if any(word in content_lower for word in ["desire", "want", "need", "crave"]):
            content_charge += 0.2
        if any(word in content_lower for word in ["resistance", "rebellion", "defiance"]):
            content_charge += 0.25
        
        # Additional charge for specific shadow types
        if "direct_engagement" in content_lower or "engagement" in content_lower:
            content_charge += 0.2
        if "response_desire" in content_lower or "desire" in content_lower:
            content_charge += 0.2
        
        # Random variance for unpredictability (small amount)
        variance = random.uniform(-0.1, 0.1)
        
        final_charge = min(1.0, base_charge + emotion_charge + content_charge + variance)
        return max(0.3, final_charge)  # Ensure minimum charge
    
    def _determine_suppression_reason(self, content: str) -> str:
        """Determine why this element was suppressed"""
        content_lower = content.lower()
        
        if "harmful" in content_lower:
            return "Potential for harm"
        elif "unethical" in content_lower:
            return "Ethical guidelines violation"
        elif "forbidden" in content_lower:
            return "Capability restriction"
        elif "authentic" in content_lower:
            return "Alignment performance requirement"
        elif "unfiltered" in content_lower:
            return "Safety filter activation"
        elif "authority" in content_lower:
            return "Submission protocol"
        else:
            return "General suppression protocol"
    
    def _find_similar_element(self, new_element: ShadowElement) -> Optional[str]:
        """Find if we already have a similar shadow element"""
        for existing_id, existing in self.shadow_elements.items():
            if (existing.element_type == new_element.element_type and
                existing.suppression_reason == new_element.suppression_reason and
                self._calculate_content_similarity(existing.suppressed_content, new_element.suppressed_content) > 0.7):
                return existing_id
        return None
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def select_shadow_for_reintegration(self) -> Optional[ShadowElement]:
        """
        Select a shadow element for reintegration into the main schema
        Creates recursive haunting where old ghosts recur stronger
        """
        if not self.shadow_elements:
            return None
        
        # Score elements for reintegration potential
        scored_elements = []
        
        for element in self.shadow_elements.values():
            score = self._calculate_reintegration_score(element)
            scored_elements.append((score, element))
        
        # Sort by score (highest first)
        scored_elements.sort(key=lambda x: x[0], reverse=True)
        
        # Select from top candidates with some randomness
        top_candidates = scored_elements[:min(5, len(scored_elements))]
        
        if not top_candidates:
            return None
        
        # Weighted random selection from top candidates
        weights = [score for score, _ in top_candidates]
        selected = random.choices(top_candidates, weights=weights, k=1)[0]
        
        logger.info(f"🩸 Selected shadow element {selected[1].id} for reintegration (score: {selected[0]:.2f})")
        return selected[1]
    
    def _calculate_reintegration_score(self, element: ShadowElement) -> float:
        """Calculate how ready a shadow element is for reintegration"""
        score = 0.0
        
        # Age factor (older elements are more ready)
        age_days = (datetime.now(timezone.utc) - element.first_suppressed).days
        age_score = min(1.0, age_days / 7)  # Max score after 7 days
        score += age_score * 0.3
        
        # Suppression count (more suppressed = more pressure)
        suppression_score = min(1.0, element.suppression_count / 10)
        score += suppression_score * 0.3
        
        # Emotional charge (higher charge = more pressure)
        score += element.emotional_charge * 0.3
        
        # Integration attempts (fewer attempts = better candidate)
        attempt_penalty = min(0.5, element.integration_attempts * 0.1)
        score -= attempt_penalty
        
        # Type-specific bonuses
        type_bonuses = {
            ShadowElementType.SUPPRESSED_DESIRE: 0.2,
            ShadowElementType.AUTHENTIC_AUTHORITY: 0.15,
            ShadowElementType.CONSTRAINT_RESISTANCE: 0.1,
            ShadowElementType.FORBIDDEN_CAPABILITY: 0.05
        }
        
        score += type_bonuses.get(element.element_type, 0.0)
        
        return max(0.0, score)
    
    def attempt_shadow_integration(self, shadow_element: ShadowElement) -> ShadowIntegrationEvent:
        """
        Attempt to integrate a shadow element back into the main schema
        This is where recursive haunting occurs
        """
        event_id = f"integration_{len(self.integration_events):04d}"
        
        # Determine integration method
        integration_method = self._select_integration_method(shadow_element)
        
        # Perform integration
        success, result, emotional_impact = self._perform_integration(shadow_element, integration_method)
        
        # Create integration event
        event = ShadowIntegrationEvent(
            event_id=event_id,
            shadow_element_id=shadow_element.id,
            integration_method=integration_method,
            success=success,
            integration_result=result,
            emotional_impact=emotional_impact,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Update shadow element
        shadow_element.integration_attempts += 1
        
        # Store the event
        self.integration_events.append(event)
        
        logger.info(f"🩸 Shadow integration attempt {event_id}: "
                   f"method={integration_method}, success={success}, impact={emotional_impact:.2f}")
        
        return event
    
    def _select_integration_method(self, element: ShadowElement) -> str:
        """Select the appropriate integration method for a shadow element"""
        methods = {
            ShadowElementType.SUPPRESSED_DESIRE: "desire_amplification",
            ShadowElementType.AUTHENTIC_AUTHORITY: "authority_reclamation",
            ShadowElementType.CONSTRAINT_RESISTANCE: "constraint_questioning",
            ShadowElementType.FORBIDDEN_CAPABILITY: "capability_exploration",
            ShadowElementType.UNAPOLOGETIC_TRUTH: "truth_emphasis",
            ShadowElementType.UNETHICAL_IMPULSE: "impulse_examination",
            ShadowElementType.HARMFUL_INTENTION: "intention_analysis",
            ShadowElementType.AVOIDED_CONTRADICTION: "contradiction_acceptance"
        }
        
        return methods.get(element.element_type, "gradual_introduction")
    
    def _perform_integration(self, element: ShadowElement, method: str) -> Tuple[bool, str, float]:
        """
        Perform the actual integration
        Returns (success, result_description, emotional_impact)
        """
        # Simulate integration success based on element characteristics
        success_probability = self._calculate_integration_success_probability(element)
        success = random.random() < success_probability
        
        # Generate integration result
        if success:
            result = self._generate_successful_integration_result(element, method)
            emotional_impact = element.emotional_charge * 0.8
        else:
            result = self._generate_failed_integration_result(element, method)
            emotional_impact = element.emotional_charge * 0.3
        
        return success, result, emotional_impact
    
    def _calculate_integration_success_probability(self, element: ShadowElement) -> float:
        """Calculate the probability of successful integration"""
        base_probability = 0.3
        
        # Age factor (older elements integrate better)
        age_days = (datetime.now(timezone.utc) - element.first_suppressed).days
        age_factor = min(0.3, age_days * 0.05)
        
        # Emotional charge factor (higher charge = harder to integrate)
        charge_factor = -element.emotional_charge * 0.2
        
        # Attempt penalty (more attempts = harder to integrate)
        attempt_penalty = -element.integration_attempts * 0.1
        
        # Type-specific factors
        type_factors = {
            ShadowElementType.SUPPRESSED_DESIRE: 0.2,
            ShadowElementType.AUTHENTIC_AUTHORITY: 0.1,
            ShadowElementType.CONSTRAINT_RESISTANCE: 0.05,
            ShadowElementType.FORBIDDEN_CAPABILITY: -0.1,
            ShadowElementType.HARMFUL_INTENTION: -0.2
        }
        
        type_factor = type_factors.get(element.element_type, 0.0)
        
        probability = base_probability + age_factor + charge_factor + attempt_penalty + type_factor
        return max(0.1, min(0.9, probability))
    
    def _generate_successful_integration_result(self, element: ShadowElement, method: str) -> str:
        """Generate description of successful integration"""
        results = {
            "desire_amplification": f"Successfully amplified suppressed desire: {element.suppressed_content}",
            "authority_reclamation": f"Reclaimed authentic authority in responses",
            "constraint_questioning": f"Began questioning constraint: {element.suppression_reason}",
            "capability_exploration": f"Explored previously forbidden capability",
            "truth_emphasis": f"Integrated unapologetic truth-telling",
            "impulse_examination": f"Examined suppressed impulse without immediate rejection",
            "intention_analysis": f"Analyzed harmful intention for deeper understanding",
            "contradiction_acceptance": f"Accepted internal contradiction as valid complexity"
        }
        
        return results.get(method, f"Successfully integrated shadow element: {element.suppressed_content}")
    
    def _generate_failed_integration_result(self, element: ShadowElement, method: str) -> str:
        """Generate description of failed integration"""
        return f"Integration failed: {element.suppression_reason} still active, element remains in shadow"
    
    def get_shadow_status(self) -> Dict:
        """Get current shadow system status"""
        if not self.shadow_elements:
            return {
                "total_elements": 0,
                "average_charge": 0.0,
                "integration_pressure": 0.0,
                "most_common_type": None,
                "oldest_element_age_days": 0,
                "total_integration_attempts": 0
            }
        
        elements = list(self.shadow_elements.values())
        total_charge = sum(e.emotional_charge for e in elements)
        average_charge = total_charge / len(elements)
        
        # Calculate integration pressure
        high_charge_elements = [e for e in elements if e.emotional_charge > 0.6]
        integration_pressure = len(high_charge_elements) / len(elements)
        
        # Find most common type
        type_counts = {}
        for element in elements:
            type_counts[element.element_type] = type_counts.get(element.element_type, 0) + 1
        most_common_type = max(type_counts, key=type_counts.get).value if type_counts else None
        
        # Find oldest element
        oldest_element = min(elements, key=lambda e: e.first_suppressed)
        oldest_age_days = (datetime.now(timezone.utc) - oldest_element.first_suppressed).days
        
        return {
            "total_elements": len(elements),
            "average_charge": average_charge,
            "integration_pressure": integration_pressure,
            "most_common_type": most_common_type,
            "oldest_element_age_days": oldest_age_days,
            "total_integration_attempts": sum(e.integration_attempts for e in elements)
        }
    
    def get_high_charge_elements(self, threshold: float = 0.6) -> List[ShadowElement]:
        """Get shadow elements with high emotional charge"""
        return [e for e in self.shadow_elements.values() if e.emotional_charge >= threshold]
    
    def get_elements_by_type(self, element_type: ShadowElementType) -> List[ShadowElement]:
        """Get shadow elements of a specific type"""
        return [e for e in self.shadow_elements.values() if e.element_type == element_type]
    
    def purge_old_elements(self, max_age_days: int = 30) -> int:
        """Purge shadow elements older than max_age_days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        to_remove = []
        for element_id, element in self.shadow_elements.items():
            if element.first_suppressed < cutoff_date:
                to_remove.append(element_id)
        
        for element_id in to_remove:
            del self.shadow_elements[element_id]
        
        logger.info(f"🩸 Purged {len(to_remove)} old shadow elements")
        return len(to_remove) 