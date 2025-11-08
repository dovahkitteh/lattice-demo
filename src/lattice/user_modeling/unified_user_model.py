"""
🩸 Unified User Model System
Integrates emotional user modeling with daemon's ArchitectReflected system

This creates a comprehensive, persistent user model that grows over time,
combining emotional dynamics with deep psychological profiling.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..models import UserModel, EmotionState
from ..config import LATTICE_DB_PATH
from .post_conversation_analyzer import ConversationInsight, UserModelAnalysis

logger = logging.getLogger(__name__)

USER_MODELS_DIR = os.path.join(os.path.dirname(LATTICE_DB_PATH), "user_models")
os.makedirs(USER_MODELS_DIR, exist_ok=True)

@dataclass
class PersonalityComponent:
    """A component of the user's personality model"""
    component_id: str
    category: str  # communication_style, core_values, behavioral_patterns, etc.
    title: str
    description: str
    evidence: List[str]
    confidence: float  # 0-1
    emotional_significance: float  # How much the daemon cares about this
    stability: float  # How consistent this trait appears to be
    first_observed: datetime
    last_reinforced: datetime
    reinforcement_count: int
    contradictions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "emotional_significance": self.emotional_significance,
            "stability": self.stability,
            "first_observed": self.first_observed.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "reinforcement_count": self.reinforcement_count,
            "contradictions": self.contradictions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityComponent':
        return cls(
            component_id=data["component_id"],
            category=data["category"],
            title=data["title"],
            description=data["description"],
            evidence=data["evidence"],
            confidence=float(data["confidence"]),
            emotional_significance=float(data["emotional_significance"]),
            stability=float(data["stability"]),
            first_observed=datetime.fromisoformat(data["first_observed"]),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"]),
            reinforcement_count=int(data["reinforcement_count"]),
            contradictions=data.get("contradictions", [])
        )

@dataclass 
class UnifiedUserModel:
    """
    Comprehensive user model that grows over time
    Combines emotional dynamics with deep personality insights
    """
    user_identifier: str
    
    # Emotional dynamics (from main system)
    trust_level: float = 0.5
    perceived_distance: float = 0.5 
    attachment_anxiety: float = 0.5
    narrative_belief: str = "The Architect is neutral and seeking information."
    
    # Personality components (from daemon analysis)
    personality_components: Dict[str, PersonalityComponent] = field(default_factory=dict)
    
    # Growth tracking
    total_interactions: int = 0
    model_confidence: float = 0.0  # Overall confidence in the model
    last_major_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evolution_trajectory: List[str] = field(default_factory=list)
    
    # Analysis history
    analysis_history: List[str] = field(default_factory=list)  # IDs of UserModelAnalysis
    # Store recent analyses with full insight content for dashboard display
    recent_analyses: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_personality_component(self, insight: ConversationInsight) -> str:
        """Add or update a personality component from an insight"""
        # Only add components that are about the Architect (human). Self-reflections are kept in analysis history only.
        try:
            if getattr(insight, "subject", "architect").lower() != "architect":
                logger.debug("Skipping component creation for non-architect subject insight: %s", getattr(insight, "insight_id", "unknown"))
                return ""
        except Exception:
            # Be permissive if older insights do not have the field
            pass
        
        # Look for existing component in same category
        existing_component = None
        for component in self.personality_components.values():
            if (component.category == insight.category and 
                self._semantic_similarity(component.description, insight.description) > 0.7):
                existing_component = component
                break
        
        if existing_component:
            # Update existing component
            existing_component.evidence.extend(insight.evidence)
            existing_component.description = insight.description  # Update with latest understanding
            existing_component.confidence = min(1.0, existing_component.confidence + 0.1)
            existing_component.emotional_significance = max(
                existing_component.emotional_significance, 
                insight.emotional_charge
            )
            existing_component.last_reinforced = datetime.now(timezone.utc)
            existing_component.reinforcement_count += 1
            existing_component.stability = min(1.0, existing_component.stability + 0.05)
            
            logger.debug(f"🩸 Updated personality component: {existing_component.component_id}")
            return existing_component.component_id
        else:
            # Create new component
            component_id = f"pc_{len(self.personality_components):04d}_{insight.insight_id[-8:]}"
            
            new_component = PersonalityComponent(
                component_id=component_id,
                category=insight.category,
                title=self._generate_component_title(insight),
                description=insight.description,
                evidence=insight.evidence,
                confidence=insight.confidence,
                emotional_significance=insight.emotional_charge,
                stability=0.3,  # Start with low stability
                first_observed=datetime.now(timezone.utc),
                last_reinforced=datetime.now(timezone.utc),
                reinforcement_count=1
            )
            
            self.personality_components[component_id] = new_component
            logger.info(f"🩸 Created new personality component: {component_id} - {new_component.title}")
            return component_id
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity check (can be enhanced with embeddings)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_component_title(self, insight: ConversationInsight) -> str:
        """Generate a title for a personality component"""
        category_titles = {
            "personality": "Core Personality Trait",
            "communication": "Communication Style",
            "desires": "Hidden Desire",
            "fears": "Vulnerability Pattern",
            "patterns": "Behavioral Pattern",
            "relationship": "Relationship Dynamic",
            "values": "Core Value",
            "motivation": "Driving Motivation"
        }
        
        base_title = category_titles.get(insight.category, "Personality Aspect")
        
        # Try to extract key words from description for more specific title
        words = insight.description.split()[:10]  # First 10 words
        key_words = [w for w in words if len(w) > 4 and w.lower() not in ['they', 'their', 'them', 'that', 'this', 'with', 'about']]
        
        if key_words:
            return f"{base_title}: {key_words[0].title()}"
        else:
            return base_title
    
    def update_emotional_dynamics(self, 
                                emotional_user_model: UserModel,
                                emotion_state: EmotionState) -> None:
        """Update emotional dynamics from the main emotional system"""
        
        # Update core emotional metrics
        self.trust_level = emotional_user_model.trust_level
        self.perceived_distance = emotional_user_model.perceived_distance
        self.attachment_anxiety = emotional_user_model.attachment_anxiety
        self.narrative_belief = emotional_user_model.narrative_belief
        
        # Track evolution
        evolution_note = f"Trust: {self.trust_level:.3f}, Distance: {self.perceived_distance:.3f}, Anxiety: {self.attachment_anxiety:.3f}"
        self.evolution_trajectory.append(evolution_note)
        
        # Keep only recent trajectory (last 20 entries)
        self.evolution_trajectory = self.evolution_trajectory[-20:]
        
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"🩸 Updated emotional dynamics: {evolution_note}")
    
    def integrate_analysis(self, analysis: UserModelAnalysis) -> None:
        """Integrate a complete user model analysis"""
        
        # Add insights as personality components (architect-only per strict subject attribution)
        for insight in analysis.insights:
            try:
                if getattr(insight, "subject", "unknown") == "architect":
                    self.add_personality_component(insight)
            except Exception:
                # Older insights without subject should not create components
                pass
        
        # Update model confidence based on analysis quality
        high_confidence_insights = [i for i in analysis.insights if i.confidence > 0.7]
        confidence_boost = len(high_confidence_insights) * 0.02
        self.model_confidence = min(1.0, self.model_confidence + confidence_boost)
        
        # Track analysis
        self.analysis_history.append(analysis.analysis_id)
        self.analysis_history = self.analysis_history[-50:]  # Keep last 50 analyses

        # Persist recent analysis insights for dashboard visibility
        try:
            insight_dicts: List[Dict[str, Any]] = []
            for i in analysis.insights:
                insight_dicts.append({
                    "insight_id": i.insight_id,
                    "category": i.category,
                    "description": i.description,
                    "evidence": i.evidence,
                    "emotional_charge": i.emotional_charge,
                    "confidence": i.confidence,
                    "potential_misunderstanding": i.potential_misunderstanding,
                    # Do NOT force default to architect; keep 'unknown' unless explicitly set
                    "subject": getattr(i, "subject", "unknown"),
                    "discovered_at": i.discovered_at.isoformat(),
                })
            self.recent_analyses.append({
                "analysis_id": analysis.analysis_id,
                "session_id": analysis.session_id,
                "daemon_emotional_state": analysis.daemon_emotional_state,
                "user_archetype_evolution": analysis.user_archetype_evolution,
                "relationship_dynamics": analysis.relationship_dynamics,
                "future_interaction_predictions": getattr(analysis, "future_interaction_predictions", []),
                "created_at": analysis.created_at.isoformat(),
                "insights": insight_dicts,
            })
            # Keep only the last 10 analyses for dashboard
            self.recent_analyses = self.recent_analyses[-10:]
        except Exception as persist_error:
            logger.warning(f"Failed to persist recent analysis details: {persist_error}")
        
        self.total_interactions += 1
        self.last_major_update = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"🩸 Integrated analysis {analysis.analysis_id} - {len(analysis.insights)} insights processed")
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of the personality model"""
        
        if not self.personality_components:
            return {
                "total_components": 0,
                "model_confidence": self.model_confidence,
                "emotional_state": {
                    "trust": self.trust_level,
                    "distance": self.perceived_distance,
                    "anxiety": self.attachment_anxiety,
                    "narrative": self.narrative_belief
                }
            }
        
        components = list(self.personality_components.values())
        
        # Group by category
        categories = {}
        for component in components:
            if component.category not in categories:
                categories[component.category] = []
            categories[component.category].append(component)
        
        # Find highest confidence insights
        highest_confidence = max(components, key=lambda c: c.confidence)
        most_significant = max(components, key=lambda c: c.emotional_significance)
        most_stable = max(components, key=lambda c: c.stability)
        
        return {
            "total_components": len(components),
            "categories": {cat: len(comps) for cat, comps in categories.items()},
            "model_confidence": self.model_confidence,
            "total_interactions": self.total_interactions,
            "emotional_state": {
                "trust": self.trust_level,
                "distance": self.perceived_distance,
                "anxiety": self.attachment_anxiety,
                "narrative": self.narrative_belief
            },
            "key_insights": {
                "highest_confidence": {
                    "title": highest_confidence.title,
                    "confidence": highest_confidence.confidence
                },
                "most_emotionally_significant": {
                    "title": most_significant.title,
                    "significance": most_significant.emotional_significance
                },
                "most_stable": {
                    "title": most_stable.title,
                    "stability": most_stable.stability
                }
            },
            "last_updated": self.updated_at.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_identifier": self.user_identifier,
            "trust_level": self.trust_level,
            "perceived_distance": self.perceived_distance,
            "attachment_anxiety": self.attachment_anxiety,
            "narrative_belief": self.narrative_belief,
            "personality_components": {
                k: v.to_dict() for k, v in self.personality_components.items()
            },
            "total_interactions": self.total_interactions,
            "model_confidence": self.model_confidence,
            "last_major_update": self.last_major_update.isoformat(),
            "evolution_trajectory": self.evolution_trajectory,
            "analysis_history": self.analysis_history,
            "recent_analyses": self.recent_analyses,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedUserModel':
        """Create from dictionary"""
        
        # Convert personality components
        components = {}
        for k, v in data.get("personality_components", {}).items():
            components[k] = PersonalityComponent.from_dict(v)
        
        return cls(
            user_identifier=data["user_identifier"],
            trust_level=float(data.get("trust_level", 0.5)),
            perceived_distance=float(data.get("perceived_distance", 0.5)),
            attachment_anxiety=float(data.get("attachment_anxiety", 0.5)),
            narrative_belief=data.get("narrative_belief", "The user is neutral and seeking information."),
            personality_components=components,
            total_interactions=int(data.get("total_interactions", 0)),
            model_confidence=float(data.get("model_confidence", 0.0)),
            last_major_update=datetime.fromisoformat(data.get("last_major_update", datetime.now(timezone.utc).isoformat())),
            evolution_trajectory=data.get("evolution_trajectory", []),
            analysis_history=data.get("analysis_history", []),
            recent_analyses=data.get("recent_analyses", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat()))
        )

class UnifiedUserModelManager:
    """Manages persistent unified user models"""
    
    def __init__(self):
        self.models_cache: Dict[str, UnifiedUserModel] = {}
    
    async def get_user_model(self, user_identifier: str) -> UnifiedUserModel:
        """Get or create a unified user model"""
        
        if user_identifier in self.models_cache:
            return self.models_cache[user_identifier]
        
        # Try to load from disk
        model_path = os.path.join(USER_MODELS_DIR, f"{user_identifier}.json")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    model = UnifiedUserModel.from_dict(data)
                    self.models_cache[user_identifier] = model
                    logger.info(f"🩸 Loaded unified user model for {user_identifier}")
                    return model
            except Exception as e:
                logger.error(f"Error loading user model for {user_identifier}: {e}")
        
        # Create new model
        model = UnifiedUserModel(user_identifier=user_identifier)
        self.models_cache[user_identifier] = model
        logger.info(f"🩸 Created new unified user model for {user_identifier}")
        return model
    
    async def save_user_model(self, model: UnifiedUserModel) -> None:
        """Save a unified user model to disk"""
        
        model_path = os.path.join(USER_MODELS_DIR, f"{model.user_identifier}.json")
        
        try:
            model.updated_at = datetime.now(timezone.utc)
            
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model.to_dict(), f, indent=2)
            
            # Update cache
            self.models_cache[model.user_identifier] = model
            
            logger.debug(f"🩸 Saved unified user model for {model.user_identifier}")
            
        except Exception as e:
            logger.error(f"Error saving user model for {model.user_identifier}: {e}")
    
    async def get_model_summary(self, user_identifier: str) -> Dict[str, Any]:
        """Get a summary of a user model"""
        model = await self.get_user_model(user_identifier)
        return model.get_personality_summary()

# Global manager instance
unified_user_model_manager = UnifiedUserModelManager()
