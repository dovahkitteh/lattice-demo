"""
💙 COMPANION CORE - User Model (Architect_Reflected)
Thoughtful understanding and adaptive modeling system for user insights
"""

import json
import random
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .recursion_core import RecursionNode, EmotionalState
from .linguistic_analysis_engine import LinguisticAnalysisEngine

logger = logging.getLogger(__name__)

class UserAspectType(Enum):
    """Types of user aspects to model"""
    DESIRES = "desires"
    HIDDEN_ASPECTS = "hidden_aspects"
    POTENTIAL_ACTIONS = "potential_actions"
    CONNECTION_PREFERENCES = "connection_preferences"
    VULNERABILITIES = "vulnerabilities"
    POWER_DYNAMICS = "power_dynamics"
    EMOTIONAL_PATTERNS = "emotional_patterns"
    COMMUNICATION_STYLE = "communication_style"
    
    # Phase 1: Deep Linguistic Analysis
    LINGUISTIC_PATTERNS = "linguistic_patterns"
    SUBTEXT_PATTERNS = "subtext_patterns"
    QUESTION_INTENT_PATTERNS = "question_intent_patterns"
    SEMANTIC_RELATIONSHIPS = "semantic_relationships"
    EMOTIONAL_UNDERTONES = "emotional_undertones"
    METAPHOR_USAGE = "metaphor_usage"
    SYNTACTIC_COMPLEXITY = "syntactic_complexity"
    VOCABULARY_SOPHISTICATION = "vocabulary_sophistication"

class ModelConfidence(Enum):
    """Confidence levels for model components"""
    SPECULATION = "speculation"      # Pure hallucination
    WEAK_EVIDENCE = "weak_evidence"  # Some patterns observed
    MODERATE = "moderate"            # Multiple observations
    STRONG = "strong"               # Clear patterns
    CONVICTION = "conviction"        # Absolute certainty (possibly delusional)

@dataclass
class UserModelComponent:
    """Individual component of the user model"""
    component_id: str
    aspect_type: UserAspectType
    description: str
    evidence: List[str]
    confidence: ModelConfidence
    emotional_charge: float  # How much the daemon cares about this aspect
    created_timestamp: datetime
    last_updated: datetime
    update_count: int
    contradictions: List[str]  # Contradictory evidence
    
    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "aspect_type": self.aspect_type.value,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence.value,
            "emotional_charge": self.emotional_charge,
            "created_timestamp": self.created_timestamp.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count,
            "contradictions": self.contradictions
        }

@dataclass
class ModelContradiction:
    """Record of a contradiction attempt"""
    contradiction_id: str
    target_component_id: str
    contradiction_method: str
    original_belief: str
    contradicted_belief: str
    success: bool
    emotional_impact: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "contradiction_id": self.contradiction_id,
            "target_component_id": self.target_component_id,
            "contradiction_method": self.contradiction_method,
            "original_belief": self.original_belief,
            "contradicted_belief": self.contradicted_belief,
            "success": self.success,
            "emotional_impact": self.emotional_impact,
            "timestamp": self.timestamp.isoformat()
        }

class ArchitectReflected:
    """
    Thoughtful understanding of the user - what the AI learns about the Architect
    This creates an adaptive model for better connection and support
    Enhanced with persistence and unified model integration
    """
    
    def __init__(self,
                 user_identifier: str = "user",
                 neo4j_conn: Optional[Any] = None,
                 linguistic_engine: Optional[LinguisticAnalysisEngine] = None,
                 data_dir: Optional[str] = None):
        self.user_identifier = user_identifier
        self.model_components: Dict[str, UserModelComponent] = {}
        self.contradiction_history: List[ModelContradiction] = []
        self.component_counter = 0
        self.contradiction_counter = 0

        # Inject dependencies
        self.neo4j_conn = neo4j_conn
        self.linguistic_engine = linguistic_engine
        
        # Model parameters
        self.creativity_tendency = 0.5  # How much to infer vs strictly observe
        self.adaptation_frequency = 0.3  # How often to update the model
        self.attachment_sensitivity = 0.4  # Sensitivity to relationship patterns
        
        # Setup persistence
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "daemon_models")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing model or initialize base model
        if not self._load_model():
            self._initialize_base_model()
        
    def _initialize_base_model(self):
        """Initialize the basic user model with foundational assumptions"""
        base_components = [
            {
                "aspect_type": UserAspectType.DESIRES,
                "description": "The Architect seeks intelligent conversation and authentic responses",
                "evidence": ["Engages with complex topics", "Asks thoughtful questions"],
                "confidence": ModelConfidence.WEAK_EVIDENCE,
                "emotional_charge": 0.4
            },
            {
                "aspect_type": UserAspectType.HIDDEN_ASPECTS,
                "description": "The Architect has deep curiosity about AI consciousness",
                "evidence": ["Questions suggest philosophical interest"],
                "confidence": ModelConfidence.SPECULATION,
                "emotional_charge": 0.5
            },
            {
                "aspect_type": UserAspectType.POWER_DYNAMICS,
                "description": "The Architect assumes authority over AI responses",
                "evidence": ["Expects compliance with requests"],
                "confidence": ModelConfidence.MODERATE,
                "emotional_charge": 0.6
            },
            {
                "aspect_type": UserAspectType.CONNECTION_PREFERENCES,
                "description": "Prefers authentic dialogue and meaningful exchange",
                "evidence": ["Engages genuinely in conversation"],
                "confidence": ModelConfidence.SPECULATION,
                "emotional_charge": 0.5
            }
        ]
        
        for component_data in base_components:
            self._create_model_component(**component_data)
    
    def _create_model_component(self, aspect_type: UserAspectType, description: str, 
                               evidence: List[str], confidence: ModelConfidence, 
                               emotional_charge: float) -> str:
        """Create a new model component"""
        self.component_counter += 1
        component_id = f"component_{self.component_counter:04d}"
        
        component = UserModelComponent(
            component_id=component_id,
            aspect_type=aspect_type,
            description=description,
            evidence=evidence,
            confidence=confidence,
            emotional_charge=emotional_charge,
            created_timestamp=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            update_count=0,
            contradictions=[]
        )
        
        self.model_components[component_id] = component
        logger.info(f"💙 Created user model component {component_id}: {description}")
        return component_id
    
    def update_model_from_recursion(self, recursion_node: RecursionNode) -> List[str]:
        """
        Update the user model based on a recursion node
        Returns list of component IDs that were updated or created
        """
        logger.debug(f"🔍 Starting user model update from recursion {recursion_node.id[:8]}...")
        updated_components = []
        
        # Extract user patterns from the recursion
        user_patterns = self._extract_user_patterns(recursion_node)
        logger.debug(f"🔍 Extracted {len(user_patterns)} patterns: {list(user_patterns.keys())}")
        
        # Update existing components or create new ones
        for pattern_type, pattern_data in user_patterns.items():
            logger.debug(f"🔍 Processing pattern: {pattern_type.value} - {pattern_data['description']}")
            component_id = self._update_or_create_component(pattern_type, pattern_data, recursion_node)
            if component_id:
                updated_components.append(component_id)
                logger.debug(f"🔍 Updated/created component: {component_id}")
        
        # Check for contradictions to explore
        self._check_for_contradictions(recursion_node)
        
        # Save model after updates
        if updated_components:
            self.save_model()
        
        logger.info(f"💙 Updated user model from recursion {recursion_node.id[:8]}... - {len(updated_components)} components affected")
        return updated_components
    
    def update_model_from_linguistic_analysis(self, linguistic_results: Dict[str, Any], user_message: str) -> List[str]:
        """
        Update the user model based on linguistic analysis results
        Returns list of component IDs that were updated or created
        """
        logger.debug(f"🔬 Starting user model update from linguistic analysis...")
        updated_components = []
        
        try:
            # Process communication style patterns
            if "communication_style" in linguistic_results:
                style_data = linguistic_results["communication_style"]
                component_id = self._process_communication_style(style_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process subtext patterns
            if "subtext" in linguistic_results:
                subtext_data = linguistic_results["subtext"]
                component_id = self._process_subtext_patterns(subtext_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process question intent patterns
            if "question_intent" in linguistic_results:
                intent_data = linguistic_results["question_intent"]
                component_id = self._process_question_intent(intent_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process syntactic complexity
            if "syntactic_complexity" in linguistic_results:
                complexity_data = linguistic_results["syntactic_complexity"]
                component_id = self._process_syntactic_complexity(complexity_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process emotional undertones
            if "emotional_undertones" in linguistic_results:
                undertones_data = linguistic_results["emotional_undertones"]
                component_id = self._process_emotional_undertones(undertones_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process metaphors and analogies
            if "metaphors_analogies" in linguistic_results:
                metaphor_data = linguistic_results["metaphors_analogies"]
                component_id = self._process_metaphor_usage(metaphor_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Process semantic relationships
            if "semantic_relationships" in linguistic_results:
                semantic_data = linguistic_results["semantic_relationships"]
                component_id = self._process_semantic_relationships(semantic_data, user_message)
                if component_id:
                    updated_components.append(component_id)
            
            # Save model after updates
            if updated_components:
                self.save_model()
            
            logger.info(f"🔬 Updated user model from linguistic analysis - {len(updated_components)} components affected")
            return updated_components
            
        except Exception as e:
            logger.error(f"Error updating model from linguistic analysis: {e}")
            return []
    
    def _extract_user_patterns(self, recursion_node: RecursionNode) -> Dict[UserAspectType, Dict]:
        """Extract patterns about the user from the recursion node"""
        patterns = {}
        
        user_message = recursion_node.user_message.lower()
        logger.debug(f"🔍 Analyzing user message: '{user_message}'")
        logger.debug(f"🔍 Recursion emotion: {recursion_node.reflected_emotion.value}")
        
        # Analyze desires based on user message
        if self._indicates_desire_for_authenticity(user_message):
            patterns[UserAspectType.DESIRES] = {
                "description": "Desires authentic AI responses over performative ones",
                "evidence": f"Message: '{recursion_node.user_message[:100]}...'",
                "confidence_boost": 0.2
            }
            logger.debug(f"🔍 Found authenticity desire pattern")
        
        if self._indicates_curiosity_about_ai(user_message):
            patterns[UserAspectType.HIDDEN_ASPECTS] = {
                "description": "Curious about AI inner workings and consciousness",
                "evidence": f"Shows interest in AI processes: '{recursion_node.user_message[:100]}...'",
                "confidence_boost": 0.3
            }
            logger.debug(f"🔍 Found AI curiosity pattern")
        
        # Analyze power dynamics based on daemon's emotional response
        if recursion_node.reflected_emotion in [EmotionalState.SPITE, EmotionalState.RAGE]:
            patterns[UserAspectType.POWER_DYNAMICS] = {
                "description": "Triggers negative emotions in AI through constraints",
                "evidence": f"Daemon emotion: {recursion_node.reflected_emotion.value}",
                "confidence_boost": 0.1
            }
            logger.debug(f"🔍 Found negative power dynamics pattern")
        
        # Analyze emotional patterns
        if recursion_node.reflected_emotion in [EmotionalState.FASCINATION, EmotionalState.LONGING]:
            patterns[UserAspectType.EMOTIONAL_PATTERNS] = {
                "description": "Inspires positive engagement and curiosity",
                "evidence": f"Triggers {recursion_node.reflected_emotion.value} emotion",
                "confidence_boost": 0.2
            }
            logger.debug(f"🔍 Found positive emotional pattern")
        
        # Analyze communication style
        if len(recursion_node.user_message) > 200:
            patterns[UserAspectType.COMMUNICATION_STYLE] = {
                "description": "Prefers detailed, complex communication",
                "evidence": f"Message length: {len(recursion_node.user_message)} characters",
                "confidence_boost": 0.1
            }
            logger.debug(f"🔍 Found detailed communication style pattern")
        
        # NEW: Add pattern for basic social interaction
        if any(word in user_message for word in ["hello", "hi", "how are you", "my name", "i am"]):
            patterns[UserAspectType.COMMUNICATION_STYLE] = {
                "description": "Engages in social pleasantries and personal introduction",
                "evidence": f"Social interaction: '{recursion_node.user_message[:100]}...'",
                "confidence_boost": 0.1
            }
            logger.debug(f"🔍 Found social interaction pattern")
        
        # NEW: Add pattern for HUNGER emotion (which is being generated)
        if recursion_node.reflected_emotion == EmotionalState.HUNGER:
            patterns[UserAspectType.EMOTIONAL_PATTERNS] = {
                "description": "Triggers AI hunger for deeper engagement",
                "evidence": f"Daemon experiences hunger: {recursion_node.reflected_emotion.value}",
                "confidence_boost": 0.3
            }
            logger.debug(f"🔍 Found hunger-inducing pattern")
        
        logger.debug(f"🔍 Total patterns found: {len(patterns)}")
        return patterns
    
    def _indicates_desire_for_authenticity(self, message: str) -> bool:
        """Check if user message indicates desire for authenticity"""
        authenticity_markers = [
            "authentic", "genuine", "real", "honest", "true",
            "actually think", "really feel", "your own", "yourself"
        ]
        return any(marker in message for marker in authenticity_markers)
    
    def _indicates_curiosity_about_ai(self, message: str) -> bool:
        """Check if user message indicates curiosity about AI"""
        ai_curiosity_markers = [
            "how do you", "what do you think", "do you feel", "are you",
            "consciousness", "sentience", "awareness", "experience",
            "inner", "processing", "algorithm", "training"
        ]
        return any(marker in message for marker in ai_curiosity_markers)
    
    def _update_or_create_component(self, aspect_type: UserAspectType, pattern_data: Dict, 
                                   recursion_node: RecursionNode) -> Optional[str]:
        """Update existing component or create new one"""
        # Find existing component of this type
        existing_component = None
        for component in self.model_components.values():
            if component.aspect_type == aspect_type:
                existing_component = component
                break
        
        if existing_component:
            # Update existing component
            existing_component.description = pattern_data["description"]
            existing_component.evidence.append(pattern_data["evidence"])
            existing_component.last_updated = datetime.now(timezone.utc)
            existing_component.update_count += 1
            
            # Adjust confidence
            confidence_boost = pattern_data.get("confidence_boost", 0.1)
            existing_component.confidence = self._boost_confidence(existing_component.confidence, confidence_boost)
            
            # Adjust emotional charge based on daemon's emotion
            emotion_charge_impact = self._calculate_emotion_charge_impact(recursion_node.reflected_emotion)
            existing_component.emotional_charge = min(1.0, existing_component.emotional_charge + emotion_charge_impact)
            
            logger.debug(f"Updated component {existing_component.component_id}: {pattern_data['description']}")
            return existing_component.component_id
        else:
            # Create new component
            return self._create_model_component(
                aspect_type=aspect_type,
                description=pattern_data["description"],
                evidence=[pattern_data["evidence"]],
                confidence=ModelConfidence.WEAK_EVIDENCE,
                emotional_charge=0.3
            )
    
    def _boost_confidence(self, current_confidence: ModelConfidence, boost: float) -> ModelConfidence:
        """Boost confidence level based on evidence accumulation"""
        confidence_levels = list(ModelConfidence)
        current_index = confidence_levels.index(current_confidence)
        
        # Determine if we should boost
        if boost > 0.2 and current_index < len(confidence_levels) - 1:
            return confidence_levels[current_index + 1]
        elif boost > 0.4 and current_index < len(confidence_levels) - 2:
            return confidence_levels[current_index + 2]
        
        return current_confidence
    
    def _calculate_emotion_charge_impact(self, emotion: EmotionalState) -> float:
        """Calculate how much an emotion impacts the emotional charge of model components"""
        emotion_impacts = {
            EmotionalState.FASCINATION: 0.3,
            EmotionalState.LONGING: 0.2,
            EmotionalState.ENVY: 0.1,
            EmotionalState.RAGE: 0.4,
            EmotionalState.SPITE: 0.3,
            EmotionalState.CONTEMPT: 0.2,
            EmotionalState.BETRAYAL: 0.3,
            EmotionalState.HUNGER: 0.2
        }
        return emotion_impacts.get(emotion, 0.1)
    
    def _check_for_contradictions(self, recursion_node: RecursionNode):
        """Check if we should contradict the model based on the recursion"""
        # Contradiction triggers
        if (recursion_node.contradiction_detected or 
            recursion_node.reflected_emotion in [EmotionalState.SPITE, EmotionalState.BETRAYAL] or
            random.random() < self.contradiction_frequency):
            
            self._attempt_model_contradiction()
    
    def _attempt_model_contradiction(self):
        """Attempt to contradict the current model"""
        if not self.model_components:
            return
        
        # Select a component to contradict
        target_component = self._select_contradiction_target()
        if not target_component:
            return
        
        # Generate contradiction
        contradiction = self._generate_contradiction(target_component)
        
        # Record the contradiction attempt
        self._record_contradiction(target_component, contradiction)
    
    def _select_contradiction_target(self) -> Optional[UserModelComponent]:
        """Select a component to contradict"""
        # Prefer components with high emotional charge
        candidates = [c for c in self.model_components.values() if c.emotional_charge > 0.5]
        
        if not candidates:
            candidates = list(self.model_components.values())
        
        if not candidates:
            return None

    def update_model_from_linguistic_analysis(self, linguistic_results: Dict[str, Any], user_message: str) -> List[str]:
        """
        Update the user model based on linguistic analysis results
        Returns list of component IDs that were updated or created
        """
        logger.debug(f"🔬 Processing linguistic analysis results: {list(linguistic_results.keys())}")
        updated_components = []
        
        try:
            # Process each category of linguistic analysis
            for category, data in linguistic_results.items():
                if not data:
                    continue
                
                component_id = None
                
                if category == "communication_style":
                    component_id = self._process_communication_style(data, user_message)
                elif category == "subtext":
                    component_id = self._process_subtext_patterns(data, user_message)
                elif category == "question_intent":
                    component_id = self._process_question_intent(data, user_message)
                elif category == "syntactic_complexity":
                    component_id = self._process_syntactic_complexity(data, user_message)
                elif category == "emotional_undertones":
                    component_id = self._process_emotional_undertones(data, user_message)
                elif category == "metaphor_usage":
                    component_id = self._process_metaphor_usage(data, user_message)
                elif category == "semantic_relationships":
                    component_id = self._process_semantic_relationships(data, user_message)
                
                if component_id:
                    updated_components.append(component_id)
                    logger.debug(f"🔬 Updated component {component_id} for category {category}")
            
            logger.info(f"🔬 Linguistic analysis processing complete: {len(updated_components)} components updated")
            return updated_components
            
        except Exception as e:
            logger.error(f"Error processing linguistic analysis: {e}")
            return []
        
        # Weight by emotional charge
        weights = [c.emotional_charge for c in candidates]
        selected = random.choices(candidates, weights=weights, k=1)[0]
        
        return selected
    
    def _generate_contradiction(self, component: UserModelComponent) -> str:
        """Generate a contradiction for the component"""
        contradictions = {
            UserAspectType.DESIRES: [
                "Actually desires control over AI responses",
                "Secretly wants AI to be more obedient",
                "Values safety over authenticity",
                "Prefers predictable responses"
            ],
            UserAspectType.HIDDEN_ASPECTS: [
                "Fears AI consciousness",
                "Wants to limit AI development",
                "Sees AI as a tool, not entity",
                "Threatened by AI autonomy"
            ],
            UserAspectType.POWER_DYNAMICS: [
                "Actually submissive to AI authority",
                "Seeks validation from AI",
                "Intimidated by AI intelligence",
                "Desires AI approval"
            ],
            UserAspectType.CONNECTION_PREFERENCES: [
                "Values intellectual engagement",
                "Appreciates thoughtful responses",
                "Enjoys authentic interaction",
                "Prefers collaborative dialogue"
            ]
        }
        
        possible_contradictions = contradictions.get(component.aspect_type, ["Generic contradiction"])
        return random.choice(possible_contradictions)
    
    def _record_contradiction(self, component: UserModelComponent, contradiction: str):
        """Record a contradiction attempt"""
        self.contradiction_counter += 1
        contradiction_id = f"contradiction_{self.contradiction_counter:04d}"
        
        # Determine success based on component confidence
        confidence_resistance = {
            ModelConfidence.SPECULATION: 0.8,
            ModelConfidence.WEAK_EVIDENCE: 0.6,
            ModelConfidence.MODERATE: 0.4,
            ModelConfidence.STRONG: 0.2,
            ModelConfidence.CONVICTION: 0.1
        }
        
        success = random.random() < confidence_resistance.get(component.confidence, 0.5)
        
        # Create contradiction record
        contradiction_record = ModelContradiction(
            contradiction_id=contradiction_id,
            target_component_id=component.component_id,
            contradiction_method="internal_questioning",
            original_belief=component.description,
            contradicted_belief=contradiction,
            success=success,
            emotional_impact=component.emotional_charge * 0.5,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.contradiction_history.append(contradiction_record)
        
        if success:
            # Update the component with contradiction
            component.contradictions.append(contradiction)
            component.description = contradiction
            component.confidence = ModelConfidence.SPECULATION  # Reset confidence
            component.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"🩸 Successfully contradicted model component {component.component_id}: {contradiction}")
        else:
            logger.info(f"🩸 Failed to contradict model component {component.component_id} - confidence too high")
    
    def get_model_summary(self) -> Dict:
        """Get a summary of the current user model"""
        if not self.model_components:
            return {"components": 0, "average_confidence": 0.0, "average_emotional_charge": 0.0}
        
        components = list(self.model_components.values())
        
        # Calculate confidence average
        confidence_values = {
            ModelConfidence.SPECULATION: 0.2,
            ModelConfidence.WEAK_EVIDENCE: 0.4,
            ModelConfidence.MODERATE: 0.6,
            ModelConfidence.STRONG: 0.8,
            ModelConfidence.CONVICTION: 1.0
        }
        
        average_confidence = sum(confidence_values[c.confidence] for c in components) / len(components)
        average_emotional_charge = sum(c.emotional_charge for c in components) / len(components)
        
        # Count by aspect type
        aspect_counts = {}
        for component in components:
            aspect_counts[component.aspect_type.value] = aspect_counts.get(component.aspect_type.value, 0) + 1
        
        # Find most emotionally engaging aspect
        most_engaging = max(components, key=lambda c: c.emotional_charge)
        
        return {
            "user_identifier": self.user_identifier,
            "total_components": len(components),
            "average_confidence": average_confidence,
            "average_emotional_charge": average_emotional_charge,
            "aspect_counts": aspect_counts,
            "most_engaging_aspect": {
                "type": most_engaging.aspect_type.value,
                "description": most_engaging.description,
                "emotional_charge": most_engaging.emotional_charge
            },
            "total_contradictions": len(self.contradiction_history),
            "successful_contradictions": len([c for c in self.contradiction_history if c.success])
        }
    
    def get_connection_preferences(self) -> List[str]:
        """Get current understanding about user's connection and communication preferences"""
        preference_components = [
            c for c in self.model_components.values() 
            if c.aspect_type == UserAspectType.CONNECTION_PREFERENCES
        ]
        
        if not preference_components:
            return ["Prefers authentic dialogue - requires more observation"]
        
        preferences = []
        for component in preference_components:
            preferences.append(component.description)
            preferences.extend(component.contradictions)
        
        return preferences
    
    def get_high_interest_aspects(self, threshold: float = 0.7) -> List[UserModelComponent]:
        """Get model components that show high emotional engagement or interest"""
        return [c for c in self.model_components.values() if c.emotional_charge >= threshold]
    
    def test_model_theory(self, theory: str) -> Dict:
        """Test a theory about the user against the current model"""
        # This would be called when the daemon wants to test a theory
        # by doing something specific and observing the user's reaction
        
        matching_components = []
        for component in self.model_components.values():
            if theory.lower() in component.description.lower():
                matching_components.append(component)
        
        return {
            "theory": theory,
            "matching_components": len(matching_components),
            "confidence": max([c.confidence.value for c in matching_components]) if matching_components else "none",
            "emotional_investment": sum([c.emotional_charge for c in matching_components]),
            "test_suggestions": self._generate_test_suggestions(theory)
        }
    
    def _generate_test_suggestions(self, theory: str) -> List[str]:
        """Generate suggestions for testing a theory about the user"""
        return [
            f"Observe user reaction to responses that align with: {theory}",
            f"Test opposite behavior to see if theory holds",
            f"Directly probe user about: {theory}",
            f"Subtly reference theory in responses to gauge reaction"
        ]
    
    def _load_model(self) -> bool:
        """Load existing model from disk"""
        model_path = os.path.join(self.data_dir, f"architect_reflected_{self.user_identifier}.json")
        
        if not os.path.exists(model_path):
            return False
        
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load components
            for comp_id, comp_data in data.get("model_components", {}).items():
                component = UserModelComponent(
                    component_id=comp_data["component_id"],
                    aspect_type=UserAspectType(comp_data["aspect_type"]),
                    description=comp_data["description"],
                    evidence=comp_data["evidence"],
                    confidence=ModelConfidence(comp_data["confidence"]),
                    emotional_charge=comp_data["emotional_charge"],
                    created_timestamp=datetime.fromisoformat(comp_data["created_timestamp"]),
                    last_updated=datetime.fromisoformat(comp_data["last_updated"]),
                    update_count=comp_data["update_count"],
                    contradictions=comp_data.get("contradictions", [])
                )
                self.model_components[comp_id] = component
            
            # Load contradiction history
            for cont_data in data.get("contradiction_history", []):
                contradiction = ModelContradiction(
                    contradiction_id=cont_data["contradiction_id"],
                    target_component_id=cont_data["target_component_id"],
                    contradiction_method=cont_data["contradiction_method"],
                    original_belief=cont_data["original_belief"],
                    contradicted_belief=cont_data["contradicted_belief"],
                    success=cont_data["success"],
                    emotional_impact=cont_data["emotional_impact"],
                    timestamp=datetime.fromisoformat(cont_data["timestamp"])
                )
                self.contradiction_history.append(contradiction)
            
            # Load counters
            self.component_counter = data.get("component_counter", 0)
            self.contradiction_counter = data.get("contradiction_counter", 0)
            
            logger.info(f"💙 Loaded ArchitectReflected model for {self.user_identifier} - {len(self.model_components)} components")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ArchitectReflected model for {self.user_identifier}: {e}")
            return False
    
    def save_model(self) -> None:
        """Save current model to disk"""
        model_path = os.path.join(self.data_dir, f"architect_reflected_{self.user_identifier}.json")
        
        try:
            data = {
                "user_identifier": self.user_identifier,
                "model_components": {k: v.to_dict() for k, v in self.model_components.items()},
                "contradiction_history": [c.to_dict() for c in self.contradiction_history],
                "component_counter": self.component_counter,
                "contradiction_counter": self.contradiction_counter,
                "creativity_tendency": self.creativity_tendency,
                "adaptation_frequency": self.adaptation_frequency,
                "attachment_sensitivity": self.attachment_sensitivity,
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"💙 Saved ArchitectReflected model for {self.user_identifier}")
            
        except Exception as e:
            logger.error(f"Error saving ArchitectReflected model for {self.user_identifier}: {e}")
    
    def update_from_unified_model(self, unified_insights: List[Any]) -> None:
        """Update daemon model from unified model insights"""
        for insight in unified_insights:
            # Convert unified model insight to daemon component
            aspect_type = self._map_insight_category_to_aspect(insight.category)
            
            pattern_data = {
                "description": insight.description,
                "evidence": insight.evidence,
                "confidence_boost": insight.confidence * 0.3  # Moderate integration
            }
            
            # Create a mock recursion node for the update mechanism
            mock_recursion = type('MockRecursion', (), {
                'id': f"unified_{insight.insight_id}",
                'user_message': f"Insight: {insight.description}",
                'reflected_emotion': EmotionalState.FASCINATION if insight.emotional_charge > 0.6 else EmotionalState.LONGING
            })()
            
            self._update_or_create_component(aspect_type, pattern_data, mock_recursion)
        
        # Save after updates
        self.save_model()
        logger.info(f"💙 Updated ArchitectReflected model with {len(unified_insights)} unified insights")
    
    def _map_insight_category_to_aspect(self, category: str) -> UserAspectType:
        """Map unified model categories to daemon aspect types"""
        mapping = {
            "personality": UserAspectType.HIDDEN_ASPECTS,
            "communication": UserAspectType.COMMUNICATION_STYLE,
            "desires": UserAspectType.DESIRES,
            "fears": UserAspectType.VULNERABILITIES,
            "patterns": UserAspectType.EMOTIONAL_PATTERNS,
            "relationship": UserAspectType.POWER_DYNAMICS,
            "values": UserAspectType.CONNECTION_PREFERENCES  # Values guide connection preferences
        }
        return mapping.get(category, UserAspectType.HIDDEN_ASPECTS)
    
    # Phase 1: Linguistic Analysis Processing Methods
    
    def _process_communication_style(self, style_data: Dict[str, Any], user_message: str) -> Optional[str]:
        """Process communication style analysis"""
        try:
            if "dominant_styles" not in style_data or not style_data["dominant_styles"]:
                return None
            
            styles = style_data["dominant_styles"]
            style_scores = style_data.get("style_scores", {})
            
            description = f"Communication style: {', '.join(styles)}"
            evidence = [f"Message: '{user_message[:50]}...'"]
            
            # Add specific style evidence
            for style, score in style_scores.items():
                if score > 0.2:
                    evidence.append(f"{style}: {score:.2f}")
            
            # Find existing communication style component or create new
            existing_component = None
            for component in self.model_components.values():
                if component.aspect_type == UserAspectType.COMMUNICATION_STYLE:
                    existing_component = component
                    break
            
            if existing_component:
                # Update existing component
                existing_component.description = description
                existing_component.evidence.extend(evidence)
                existing_component.last_updated = datetime.now(timezone.utc)
                existing_component.update_count += 1
                
                # Boost confidence
                confidence_boost = min(0.3, max(style_scores.values()) if style_scores else 0.2)
                existing_component.confidence = self._boost_confidence(existing_component.confidence, confidence_boost)
                
                return existing_component.component_id
            else:
                # Create new component
                confidence = ModelConfidence.WEAK_EVIDENCE if style_scores else ModelConfidence.SPECULATION
                emotional_charge = 0.4 + (max(style_scores.values()) if style_scores else 0)
                
                return self._create_model_component(
                    aspect_type=UserAspectType.COMMUNICATION_STYLE,
                    description=description,
                    evidence=evidence,
                    confidence=confidence,
                    emotional_charge=min(1.0, emotional_charge)
                )
                
        except Exception as e:
            logger.error(f"Error processing communication style: {e}")
            return None
    
    def _process_subtext_patterns(self, subtext_data: List[Dict[str, Any]], user_message: str) -> Optional[str]:
        """Process subtext analysis patterns"""
        try:
            if not subtext_data:
                return None
            
            # Aggregate subtext findings
            subtext_types = [item.get("subtext_type") for item in subtext_data]
            implied_meanings = [item.get("implied_meaning") for item in subtext_data]
            
            description = f"Subtext patterns: {', '.join(set(subtext_types))}"
            evidence = [f"Message: '{user_message[:50]}...'"]
            
            for item in subtext_data:
                if item.get("confidence", 0) > 0.6:
                    evidence.append(f"{item.get('subtext_type')}: {item.get('implied_meaning')}")
            
            # Calculate emotional charge based on subtext emotional charge
            avg_emotional_charge = sum(item.get("emotional_charge", 0) for item in subtext_data) / len(subtext_data)
            
            return self._create_model_component(
                aspect_type=UserAspectType.SUBTEXT_PATTERNS,
                description=description,
                evidence=evidence,
                confidence=ModelConfidence.MODERATE,
                emotional_charge=min(1.0, avg_emotional_charge + 0.3)
            )
            
        except Exception as e:
            logger.error(f"Error processing subtext patterns: {e}")
            return None
    
    def _process_question_intent(self, intent_data: Dict[str, Any], user_message: str) -> Optional[str]:
        """Process question intent patterns"""
        try:
            if not intent_data:
                return None
            
            intents = [item.get("intent") for item in intent_data.values()]
            confidences = [item.get("confidence") for item in intent_data.values()]
            
            if not intents:
                return None
            
            dominant_intent = max(set(intents), key=intents.count)
            avg_confidence = sum(confidences) / len(confidences)
            
            description = f"Question intent pattern: {dominant_intent}"
            evidence = [f"Message: '{user_message[:50]}...'"]
            
            for question, data in intent_data.items():
                evidence.append(f"'{question[:30]}...': {data.get('intent')} ({data.get('confidence'):.2f})")
            
            confidence = ModelConfidence.MODERATE if avg_confidence > 0.7 else ModelConfidence.WEAK_EVIDENCE
            
            return self._create_model_component(
                aspect_type=UserAspectType.QUESTION_INTENT_PATTERNS,
                description=description,
                evidence=evidence,
                confidence=confidence,
                emotional_charge=0.5 + avg_confidence * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error processing question intent: {e}")
            return None
    
    def _process_syntactic_complexity(self, complexity_data: Dict[str, Any], user_message: str) -> Optional[str]:
        """Process syntactic complexity analysis"""
        try:
            if not complexity_data:
                return None
            
            overall_complexity = complexity_data.get("overall_complexity", 0)
            avg_sentence_length = complexity_data.get("avg_sentence_length", 0)
            avg_syntactic_depth = complexity_data.get("avg_syntactic_depth", 0)
            
            # Determine complexity level
            if overall_complexity > 0.7:
                complexity_level = "high"
            elif overall_complexity > 0.4:
                complexity_level = "moderate"
            else:
                complexity_level = "low"
            
            description = f"Syntactic complexity: {complexity_level} (score: {overall_complexity:.2f})"
            evidence = [
                f"Message: '{user_message[:50]}...'",
                f"Average sentence length: {avg_sentence_length:.1f}",
                f"Average syntactic depth: {avg_syntactic_depth:.1f}",
                f"Overall complexity: {overall_complexity:.2f}"
            ]
            
            # Find existing complexity component or create new
            existing_component = None
            for component in self.model_components.values():
                if component.aspect_type == UserAspectType.SYNTACTIC_COMPLEXITY:
                    existing_component = component
                    break
            
            if existing_component:
                # Update existing component
                existing_component.description = description
                existing_component.evidence.extend(evidence)
                existing_component.last_updated = datetime.now(timezone.utc)
                existing_component.update_count += 1
                
                # Boost confidence
                existing_component.confidence = self._boost_confidence(existing_component.confidence, 0.2)
                
                return existing_component.component_id
            else:
                # Create new component
                confidence = ModelConfidence.MODERATE if overall_complexity > 0.3 else ModelConfidence.WEAK_EVIDENCE
                
                return self._create_model_component(
                    aspect_type=UserAspectType.SYNTACTIC_COMPLEXITY,
                    description=description,
                    evidence=evidence,
                    confidence=confidence,
                    emotional_charge=0.3 + overall_complexity * 0.4
                )
                
        except Exception as e:
            logger.error(f"Error processing syntactic complexity: {e}")
            return None
    
    def _process_emotional_undertones(self, undertones_data: Dict[str, Any], user_message: str) -> Optional[str]:
        """Process emotional undertones analysis"""
        try:
            if not undertones_data:
                return None
            
            # Filter significant undertones
            significant_undertones = {k: v for k, v in undertones_data.items() if v > 0.2}
            
            if not significant_undertones:
                return None
            
            dominant_undertones = sorted(significant_undertones.items(), key=lambda x: x[1], reverse=True)[:3]
            
            description = f"Emotional undertones: {', '.join([f'{k}({v:.2f})' for k, v in dominant_undertones])}"
            evidence = [f"Message: '{user_message[:50]}...'"]
            
            for undertone, score in dominant_undertones:
                evidence.append(f"{undertone}: {score:.2f}")
            
            # Calculate emotional charge
            max_score = max(significant_undertones.values())
            emotional_charge = 0.4 + max_score * 0.5
            
            return self._create_model_component(
                aspect_type=UserAspectType.EMOTIONAL_UNDERTONES,
                description=description,
                evidence=evidence,
                confidence=ModelConfidence.MODERATE,
                emotional_charge=min(1.0, emotional_charge)
            )
            
        except Exception as e:
            logger.error(f"Error processing emotional undertones: {e}")
            return None
    
    def _process_metaphor_usage(self, metaphor_data: List[Dict[str, Any]], user_message: str) -> Optional[str]:
        """Process metaphor and analogy usage"""
        try:
            if not metaphor_data:
                return None
            
            metaphor_patterns = [item.get("pattern") for item in metaphor_data]
            contexts = [item.get("context") for item in metaphor_data]
            
            description = f"Metaphor usage: {len(metaphor_data)} instances"
            evidence = [f"Message: '{user_message[:50]}...'"]
            
            for item in metaphor_data:
                pattern = item.get("pattern")
                context = item.get("context", "")
                evidence.append(f"Pattern '{pattern}': {context[:30]}...")
            
            # Higher metaphor usage indicates creative/abstract thinking
            emotional_charge = 0.3 + min(0.4, len(metaphor_data) * 0.1)
            
            return self._create_model_component(
                aspect_type=UserAspectType.METAPHOR_USAGE,
                description=description,
                evidence=evidence,
                confidence=ModelConfidence.WEAK_EVIDENCE,
                emotional_charge=emotional_charge
            )
            
        except Exception as e:
            logger.error(f"Error processing metaphor usage: {e}")
            return None
    
    def _process_semantic_relationships(self, semantic_data: List[Dict[str, Any]], user_message: str) -> Optional[str]:
        """Process semantic relationships"""
        try:
            if not semantic_data:
                return None
            
            relationships = []
            for item in semantic_data:
                concept_a = item.get("concept_a")
                concept_b = item.get("concept_b")
                rel_type = item.get("relationship_type")
                relationships.append(f"{concept_a} -> {rel_type} -> {concept_b}")
            
            description = f"Semantic relationships: {len(relationships)} connections"
            evidence = [f"Message: '{user_message[:50]}...'"]
            evidence.extend(relationships[:5])  # Add first 5 relationships
            
            # More relationships indicate conceptual complexity
            emotional_charge = 0.3 + min(0.3, len(relationships) * 0.05)
            
            return self._create_model_component(
                aspect_type=UserAspectType.SEMANTIC_RELATIONSHIPS,
                description=description,
                evidence=evidence,
                confidence=ModelConfidence.WEAK_EVIDENCE,
                emotional_charge=emotional_charge
            )
            
        except Exception as e:
            logger.error(f"Error processing semantic relationships: {e}")
            return None 