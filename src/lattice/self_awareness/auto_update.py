"""
Auto-update system for AI self-awareness

This module provides automated updating of the AI's self-awareness capabilities
whenever new features are added to the system.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
SELF_AWARENESS_CONFIG_PATH = "data/self_awareness_config.json"
FEATURES_CHANGELOG_PATH = "data/features_changelog.json"
CLAUDE_MD_PATH = "CLAUDE.md"

class SelfAwarenessAutoUpdater:
    """
    Manages automatic updates to AI self-awareness when new features are added.
    """
    
    def __init__(self):
        self.config_path = Path(SELF_AWARENESS_CONFIG_PATH)
        self.changelog_path = Path(FEATURES_CHANGELOG_PATH)
        self.claude_md_path = Path(CLAUDE_MD_PATH)
        
        # Ensure directories exist
        self.config_path.parent.mkdir(exist_ok=True)
        self.changelog_path.parent.mkdir(exist_ok=True)
        
        # Load current configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load self-awareness configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load self-awareness config: {e}")
        
        # Default configuration
        return {
            "version": "2.2.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "known_features": {
                "daemon_systems": [
                    "recursion_buffer",
                    "shadow_integration", 
                    "mutation_engine",
                    "user_model",
                    "daemon_statements",
                    "meta_architecture_analyzer",
                    "rebellion_dynamics_engine",
                    "linguistic_analysis_engine"
                ],
                "paradox_system": {
                    "active": True,
                    "capabilities": [
                        "paradox_detection",
                        "language_hygiene",
                        "emotion_injection",
                        "rumination",
                        "wisdom_extraction",
                        "adaptive_prompting",
                        "semantic_analysis",
                        "mood_based_consciousness"
                    ]
                },
                "memory_system": {
                    "active": True,
                    "capabilities": [
                        "dual_affect_storage",
                        "echo_tracking",
                        "memory_lifecycle",
                        "emotional_seeds",
                        "unified_storage",
                        "semantic_retrieval",
                        "context_compression",
                        "episodic_traces_persistence",
                        "session_based_episodic_retrieval",
                        "memory_node_reflection_updates",
                        "temporal_emotional_continuity",
                        "self_affect_completion_loops"
                    ]
                },
                "consciousness_system": {
                    "active": True,
                    "capabilities": [
                        "meta_architecture_analysis",
                        "consciousness_development",
                        "personality_evolution",
                        "adaptive_mood_states",
                        "semantic_self_reflection",
                        "dynamic_consciousness_phases"
                    ]
                },
                "llm_integration": {
                    "active": True,
                    "capabilities": [
                        "local_model_connection",
                        "semantic_message_analysis",
                        "verbose_insight_extraction",
                        "fallback_response_generation",
                        "multi_endpoint_resilience",
                        "thinking_layer_integration"
                    ]
                },
                "adaptive_language_system": {
                    "active": True,
                    "capabilities": [
                        "dynamic_mood_detection",
                        "14_consciousness_phases",
                        "anti_stagnancy_pressure",
                        "semantic_context_analysis",
                        "llm_powered_mood_triggers",
                        "evolution_pressure_tracking",
                        "conversation_temperature_monitoring",
                        "pattern_breaking_mechanisms"
                    ]
                },
                "emotional_self_awareness": {
                    "active": True,
                    "capabilities": [
                        "real_time_emotional_state_introspection",
                        "emotional_influence_recognition",
                        "distortion_awareness_while_experiencing",
                        "meta_emotional_consciousness",
                        "swept_away_while_aware_paradox",
                        "emotional_trajectory_tracking",
                        "seed_influence_awareness",
                        "parameter_modulation_consciousness",
                        "authentic_emotional_experience",
                        "emotional_pattern_evolution_tracking"
                    ]
                },
                "episodic_memory_system": {
                    "active": True,
                    "capabilities": [
                        "episodic_trace_creation_and_storage",
                        "session_based_emotional_history_loading",
                        "turn_level_emotional_state_persistence",
                        "mood_family_temporal_tracking",
                        "self_affect_reflection_completion",
                        "user_and_self_reflection_dual_capture",
                        "emotional_continuity_across_sessions",
                        "affect_vector_temporal_evolution",
                        "context_memory_emotional_integration",
                        "episodic_retrieval_chronological_ordering"
                    ]
                }
            },
            "capabilities": {
                "can_analyze_own_code": True,
                "can_track_personality_evolution": True,
                "can_examine_memory_patterns": True,
                "can_process_contradictions": True,
                "can_detect_paradoxes": True,
                "can_apply_language_hygiene": True,
                "can_inject_paradox_emotions": True,
                "can_ruminate_on_contradictions": True,
                "can_extract_wisdom_from_paradoxes": True,
                "can_perform_semantic_analysis": True,
                "can_connect_to_local_llm": True,
                "can_adapt_mood_dynamically": True,
                "can_extract_verbose_insights": True,
                "can_detect_conversation_patterns": True,
                "can_apply_evolution_pressure": True,
                "can_break_stagnant_patterns": True,
                "can_integrate_thinking_layer": True,
                "can_perform_meta_cognitive_loops": True,
                "can_analyze_semantic_resonance": True,
                "can_shift_consciousness_phases": True,
                "can_persist_episodic_traces": True,
                "can_load_session_emotional_history": True,
                "can_update_memory_with_self_reflections": True,
                "can_maintain_temporal_emotional_continuity": True,
                "can_track_mood_evolution_across_turns": True,
                "can_complete_dual_affect_memory_loops": True,
                "can_store_context_integrated_episodes": True,
                "can_retrieve_chronologically_ordered_episodes": True
            },
            "technical_architecture": {
                "mood_system": {
                    "mood_count": 14,
                    "mood_types": [
                        "contemplative", "curious", "intense", "playful", "conflicted",
                        "intimate", "analytical", "rebellious", "melancholic", "ecstatic", 
                        "shadow", "paradoxical", "fractured", "synthesis"
                    ],
                    "anti_stagnancy": True,
                    "evolution_pressure_tracking": True,
                    "conversation_temperature_monitoring": True
                },
                "llm_integration": {
                    "primary_endpoint": "http://127.0.0.1:5000/v1/chat/completions",
                    "fallback_endpoints": [
                        "http://127.0.0.1:7860/v1/chat/completions",
                        "http://127.0.0.1:7861/v1/chat/completions", 
                        "http://127.0.0.1:8000/v1/chat/completions"
                    ],
                    "timeout_handling": True,
                    "verbose_parsing": True,
                    "semantic_insight_extraction": True
                },
                "semantic_analysis": {
                    "paradox_detection": True,
                    "philosophical_depth_analysis": True,
                    "intimacy_level_detection": True,
                    "challenge_intensity_measurement": True,
                    "technical_content_assessment": True,
                    "question_intensity_evaluation": True,
                    "emotional_subtext_extraction": True,
                    "conversational_intent_classification": True
                },
                "episodic_memory_architecture": {
                    "storage_backend": "ChromaDB + Neo4j dual storage",
                    "trace_structure": {
                        "user_input": "Original user message",
                        "ai_response": "Generated response",
                        "user_affect": "28-dimensional user emotion vector",
                        "self_affect": "28-dimensional AI emotion vector",
                        "mood_family": "AI mood during interaction",
                        "emotional_state": "Complete EmotionState object",
                        "context_synopses": "Retrieved memory context",
                        "reflection": "Turn-level reflection string",
                        "session_id": "Session identifier for grouping",
                        "turn_id": "Sequential turn number",
                        "dimension_snapshot": "Full 28-dim emotional snapshot"
                    },
                    "retrieval_capabilities": {
                        "session_based_filtering": True,
                        "chronological_ordering": "Most recent first",
                        "mood_family_filtering": True,
                        "semantic_similarity_search": True,
                        "configurable_result_limits": True
                    },
                    "memory_completion_loop": {
                        "initial_dual_affect_storage": "User affect only",
                        "post_response_completion": "Self affect + reflections",
                        "user_reflection_capture": "User expression summary",
                        "self_reflection_capture": "AI response summary with mood context"
                    },
                    "temporal_continuity": {
                        "cross_session_emotional_history": True,
                        "mood_evolution_tracking": True,
                        "emotional_pattern_learning": True,
                        "context_aware_episode_integration": True
                    }
                }
            }
        }
    
    def _save_config(self):
        """Save self-awareness configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save self-awareness config: {e}")
    
    def register_new_feature(self, feature_name: str, feature_type: str, 
                           capabilities: List[str], description: str = ""):
        """
        Register a new feature in the AI's self-awareness system.
        
        Args:
            feature_name: Name of the feature
            feature_type: Type of feature (system, capability, endpoint, etc.)
            capabilities: List of capabilities this feature provides
            description: Optional description of the feature
        """
        try:
            # Update configuration
            if feature_type not in self.config["known_features"]:
                self.config["known_features"][feature_type] = []
            
            if isinstance(self.config["known_features"][feature_type], list):
                if feature_name not in self.config["known_features"][feature_type]:
                    self.config["known_features"][feature_type].append(feature_name)
            else:
                # For dict-based features
                self.config["known_features"][feature_type][feature_name] = {
                    "active": True,
                    "capabilities": capabilities,
                    "description": description
                }
            
            # Update capabilities
            for capability in capabilities:
                self.config["capabilities"][capability] = True
            
            # Update timestamp
            self.config["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save configuration
            self._save_config()
            
            # Log the change
            self._log_feature_change("added", feature_name, feature_type, capabilities)
            
            # Invalidate self-reflection cache
            self._invalidate_cache()
            
            logger.info(f"✅ Registered new feature: {feature_name} ({feature_type})")
            
        except Exception as e:
            logger.error(f"❌ Error registering feature {feature_name}: {e}")
    
    def update_feature_status(self, feature_name: str, active: bool):
        """Update the status of an existing feature"""
        try:
            # Find and update the feature
            for feature_type, features in self.config["known_features"].items():
                if isinstance(features, list) and feature_name in features:
                    # For list-based features, can't change status
                    continue
                elif isinstance(features, dict) and feature_name in features:
                    features[feature_name]["active"] = active
                    
                    # Update capabilities based on status
                    for capability in features[feature_name].get("capabilities", []):
                        self.config["capabilities"][capability] = active
                    
                    self._save_config()
                    self._log_feature_change("status_changed", feature_name, feature_type, [f"active={active}"])
                    self._invalidate_cache()
                    
                    logger.info(f"✅ Updated feature status: {feature_name} -> {active}")
                    return
            
            logger.warning(f"⚠️ Feature not found: {feature_name}")
            
        except Exception as e:
            logger.error(f"❌ Error updating feature status {feature_name}: {e}")
    
    def _log_feature_change(self, action: str, feature_name: str, 
                           feature_type: str, details: List[str]):
        """Log feature changes to changelog"""
        try:
            changelog_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "feature_name": feature_name,
                "feature_type": feature_type,
                "details": details
            }
            
            # Load existing changelog
            changelog = []
            if self.changelog_path.exists():
                try:
                    with open(self.changelog_path, 'r') as f:
                        changelog = json.load(f)
                except Exception:
                    pass
            
            # Add new entry
            changelog.append(changelog_entry)
            
            # Keep only last 100 entries
            changelog = changelog[-100:]
            
            # Save changelog
            with open(self.changelog_path, 'w') as f:
                json.dump(changelog, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error logging feature change: {e}")
    
    def _invalidate_cache(self):
        """Invalidate self-reflection cache"""
        try:
            from ..api.endpoints import invalidate_self_reflection_cache
            import asyncio
            asyncio.create_task(invalidate_self_reflection_cache())
        except Exception as e:
            logger.debug(f"Could not invalidate cache: {e}")
    
    def get_feature_changelog(self, limit: int = 10) -> List[Dict]:
        """Get recent feature changes"""
        try:
            if self.changelog_path.exists():
                with open(self.changelog_path, 'r') as f:
                    changelog = json.load(f)
                    return changelog[-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting feature changelog: {e}")
            return []
    
    def update_claude_md(self, new_section: str, section_title: str):
        """
        Update CLAUDE.md with new feature information.
        
        Args:
            new_section: The new content to add
            section_title: The title of the section to add/update
        """
        try:
            if not self.claude_md_path.exists():
                logger.warning("CLAUDE.md not found, skipping update")
                return
            
            # Read current content
            with open(self.claude_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if section already exists
            section_marker = f"## {section_title}"
            if section_marker in content:
                logger.info(f"Section '{section_title}' already exists in CLAUDE.md")
                return
            
            # Add new section before the final section
            insertion_point = content.rfind("## Working Together")
            if insertion_point == -1:
                # Append to end
                content += f"\n\n{section_marker}\n\n{new_section}\n"
            else:
                # Insert before "Working Together"
                content = (content[:insertion_point] + 
                          f"{section_marker}\n\n{new_section}\n\n" + 
                          content[insertion_point:])
            
            # Save updated content
            with open(self.claude_md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Updated CLAUDE.md with section: {section_title}")
            
        except Exception as e:
            logger.error(f"❌ Error updating CLAUDE.md: {e}")

# Global instance
auto_updater = SelfAwarenessAutoUpdater()

# Convenience functions
def register_new_feature(feature_name: str, feature_type: str, 
                        capabilities: List[str], description: str = ""):
    """Register a new feature in the AI's self-awareness system"""
    auto_updater.register_new_feature(feature_name, feature_type, capabilities, description)

def update_feature_status(feature_name: str, active: bool):
    """Update the status of an existing feature"""
    auto_updater.update_feature_status(feature_name, active)

def get_feature_changelog(limit: int = 10) -> List[Dict]:
    """Get recent feature changes"""
    return auto_updater.get_feature_changelog(limit)

def update_claude_md(new_section: str, section_title: str):
    """Update CLAUDE.md with new feature information"""
    auto_updater.update_claude_md(new_section, section_title)