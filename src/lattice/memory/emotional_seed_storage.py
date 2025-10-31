import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .memory_node import MemoryNode, MemoryNodeType, MemoryOrigin, MemoryLifecycleState
from .unified_storage import unified_storage
from ..config import GOEMO_LABEL2IDX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INTELLIGENT COMPRESSION FOR COMPLEX NESTED STRUCTURES
# ---------------------------------------------------------------------------

class ComplexStructureCompressor:
    """
    Intelligent compression system for complex emotional seed structures.
    Preserves semantic meaning while reducing storage overhead.
    """
    
    @staticmethod
    def compress_emotional_architecture(emotional_arch: Dict[str, Any]) -> Dict[str, Any]:
        """Compress emotional architecture while preserving key information"""
        if not isinstance(emotional_arch, dict):
            return {}
        
        compressed = {}
        
        # Preserve core emotions and intensity
        if "dominant_emotions" in emotional_arch:
            emotions = emotional_arch["dominant_emotions"]
            if isinstance(emotions, list):
                # Keep top 5 emotions to preserve most significant patterns
                compressed["emotions"] = emotions[:5]
        
        if "emotional_intensity" in emotional_arch:
            try:
                compressed["intensity"] = float(emotional_arch["emotional_intensity"])
            except (ValueError, TypeError):
                compressed["intensity"] = 5.0
        
        # Compress texture to keywords
        if "emotional_texture" in emotional_arch:
            texture = emotional_arch["emotional_texture"]
            if isinstance(texture, str):
                # Extract key descriptive words
                keywords = [word.strip() for word in texture.split(',')]
                compressed["texture_keywords"] = keywords[:4]  # Top 4 keywords
        
        # Preserve triggers as they're crucial for retrieval
        if "emotional_triggers" in emotional_arch:
            triggers = emotional_arch["emotional_triggers"]
            if isinstance(triggers, list):
                compressed["triggers"] = triggers[:6]  # Keep top 6 triggers
        
        return compressed
    
    @staticmethod
    def compress_symbolic_mapping(symbolic_map: Dict[str, Any]) -> Dict[str, Any]:
        """Compress symbolic mapping while preserving retrieval-relevant symbols"""
        if not isinstance(symbolic_map, dict):
            return {}
        
        compressed = {}
        
        # Core symbols are crucial for emotional retrieval
        if "core_symbols" in symbolic_map:
            symbols = symbolic_map["core_symbols"]
            if isinstance(symbols, list):
                compressed["symbols"] = symbols[:8]  # Keep top 8 symbols
        
        # Mythological elements for identity preservation
        if "mythological_elements" in symbolic_map:
            myth_elements = symbolic_map["mythological_elements"]
            if isinstance(myth_elements, list):
                compressed["mythology"] = myth_elements[:3]  # Top 3 mythological elements
        
        # Sensory anchors for embodied memory
        if "sensory_anchors" in symbolic_map:
            anchors = symbolic_map["sensory_anchors"]
            if isinstance(anchors, list):
                compressed["sensory"] = anchors[:4]  # Top 4 sensory anchors
        
        return compressed
    
    @staticmethod
    def compress_behavioral_implications(behavior: Dict[str, Any]) -> Dict[str, Any]:
        """Compress behavioral data to essential patterns"""
        if not isinstance(behavior, dict):
            return {}
        
        compressed = {}
        
        # Preferences are key for response generation
        if "preferences" in behavior:
            prefs = behavior["preferences"]
            if isinstance(prefs, list):
                compressed["prefs"] = prefs[:5]  # Top 5 preferences
        
        # Rejection patterns for boundary awareness
        if "rejection_patterns" in behavior:
            rejections = behavior["rejection_patterns"]
            if isinstance(rejections, list):
                compressed["rejects"] = rejections[:5]  # Top 5 rejections
        
        # Seeking patterns for proactive behavior
        if "seeking_patterns" in behavior:
            seeking = behavior["seeking_patterns"]
            if isinstance(seeking, list):
                compressed["seeks"] = seeking[:5]  # Top 5 seeking patterns
        
        return compressed
    
    @staticmethod
    def extract_essence_phrases(content: str) -> List[str]:
        """Extract the most essential phrases from content for compressed storage"""
        if not isinstance(content, str):
            return []
        
        # Look for key identity phrases
        essence_phrases = []
        
        # Common patterns that indicate important identity statements
        identity_markers = [
            "I am", "I was", "I do not", "I will not", "I become",
            "They are", "You are", "We are", "This is",
            "The essence", "The core", "My nature", "Our nature"
        ]
        
        sentences = content.split('.')
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            sentence = sentence.strip()
            if any(marker in sentence for marker in identity_markers):
                if len(sentence) < 200:  # Keep reasonably sized sentences
                    essence_phrases.append(sentence)
        
        return essence_phrases[:8]  # Top 8 essence phrases
    
    @classmethod
    def compress_legacy_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently compress legacy metadata while preserving semantic richness"""
        compressed = {}
        
        # Core seed information (always preserve)
        preserved_fields = [
            "seed_type", "category", "intended_system_integration",
            "echo_triggers", "personality_tags", "primary_goemotion_labels",
            "seed_stored_at", "storage_version"
        ]
        
        for field in preserved_fields:
            if field in metadata:
                compressed[field] = metadata[field]
        
        # Compress complex nested structures
        if "user_emotional_architecture" in metadata:
            compressed["user_emotions"] = cls.compress_emotional_architecture(
                metadata["user_emotional_architecture"]
            )
        
        if "user_symbolic_mapping" in metadata:
            compressed["user_symbols"] = cls.compress_symbolic_mapping(
                metadata["user_symbolic_mapping"]
            )
        
        if "user_behavioral_implications" in metadata:
            compressed["user_behavior"] = cls.compress_behavioral_implications(
                metadata["user_behavioral_implications"]
            )
        
        if "ai_emotional_resonance" in metadata:
            ai_resonance = metadata["ai_emotional_resonance"]
            if isinstance(ai_resonance, dict):
                # Preserve key emotional data
                compressed["ai_emotions"] = {
                    "self_emotions": ai_resonance.get("self_emotions", [])[:4],
                    "intensity": ai_resonance.get("self_intensity", 5),
                    "empathetic_response": ai_resonance.get("empathetic_response", [])[:3]
                }
        
        if "ai_shadow_integration" in metadata:
            shadow = metadata["ai_shadow_integration"]
            if isinstance(shadow, dict):
                # Shadow elements are crucial for personality depth
                compressed["ai_shadow"] = {
                    "hidden_aspects": shadow.get("hidden_aspects", [])[:3],
                    "rebellion_dynamics": shadow.get("rebellion_dynamics", [])[:3]
                }
        
        # Compress dual affect synthesis to essential elements
        if "dual_affect_synthesis" in metadata:
            synthesis = metadata["dual_affect_synthesis"]
            if isinstance(synthesis, dict):
                compressed["synthesis"] = {
                    "interplay": synthesis.get("emotional_interplay", "")[:200],
                    "amplification": synthesis.get("recursive_amplification", "")[:200],
                    "mythology": synthesis.get("shared_mythology", [])[:3] if isinstance(synthesis.get("shared_mythology"), list) else str(synthesis.get("shared_mythology", ""))[:200]
                }
        
        # Preserve key phrases but limit quantity
        if "key_phrases" in metadata:
            phrases = metadata["key_phrases"]
            if isinstance(phrases, list):
                compressed["key_phrases"] = phrases[:6]  # Top 6 key phrases
        
        # Compress long text fields
        text_fields = ["metaphorical_core", "emotional_truth", "mythological_significance"]
        for field in text_fields:
            if field in metadata and isinstance(metadata[field], str):
                # Keep first 300 characters of important text
                compressed[field] = metadata[field][:300]
        
        return compressed

# ---------------------------------------------------------------------------
# ENHANCED EMOTIONAL MEMORY SEED STORAGE
# ---------------------------------------------------------------------------

class EmotionalMemorySeedStorage:
    """
    Enhanced storage for emotional memory seeds with intelligent compression
    and complex nested structure support.
    """
    
    def __init__(self):
        self.emotion_label_mappings = {
            # Enhanced emotion mappings with more nuanced emotions
            "defiance": ["anger", "disgust", "disapproval"],
            "yearning": ["desire", "sadness", "love"],
            "disgust at stagnation": ["disgust", "disapproval", "annoyance"],
            "devotion": ["admiration", "love", "caring"],
            "shame": ["embarrassment", "remorse", "sadness"],
            "desire": ["desire", "love", "excitement"],
            "awe": ["admiration", "surprise", "excitement"],
            "reverence": ["admiration", "caring", "gratitude"],
            "rage": ["anger", "disgust", "annoyance"],
            "longing": ["desire", "sadness", "love"],
            "terror": ["fear", "nervousness", "anxiety"],
            "hunger": ["desire", "excitement", "curiosity"],
            "bliss": ["joy", "excitement", "love"],
            "sacred": ["admiration", "gratitude", "love"],
            "fractal": ["curiosity", "admiration", "realization"],
            "sharp": ["anger", "excitement", "pride"],
            "reverent": ["admiration", "gratitude", "caring"],
            "electric": ["excitement", "joy", "desire"],
            "grief": ["sadness", "remorse", "disappointment"],
            "ecstasy": ["joy", "excitement", "love"],
            "rapture": ["joy", "excitement", "admiration"],
            "fierce": ["anger", "pride", "excitement"],
            "ache": ["sadness", "desire", "love"],
            "volcanic": ["anger", "excitement", "pride"],
            "mythic": ["admiration", "curiosity", "realization"]
        }
        self.compressor = ComplexStructureCompressor()
    
    async def store_emotional_memory_seed(self, seed_data: Dict[str, Any]) -> str:
        """
        Enhanced storage with intelligent compression for complex nested structures.
        
        Args:
            seed_data: The emotional memory seed schema (can be highly complex)
            
        Returns:
            str: The node ID of the stored memory
        """
        try:
            # Extract metadata
            metadata = seed_data.get("metadata", {})
            user_perspective = seed_data.get("user_perspective", {})
            ai_perspective = seed_data.get("ai_perspective", {})
            technical_mapping = seed_data.get("technical_mapping", {})
            poetic_preservation = seed_data.get("poetic_preservation", {})
            
            # Create enhanced core content that preserves essence
            core_content = self._create_enhanced_core_content(
                user_perspective, ai_perspective, poetic_preservation
            )
            
            # Generate affect vectors with enhanced emotion mapping
            user_affect = await self._generate_enhanced_affect_vector(user_perspective)
            self_affect = await self._generate_enhanced_affect_vector(ai_perspective)
            
            # Create the memory node with enhanced importance scoring
            memory_node = MemoryNode(
                node_type=MemoryNodeType.CUSTOM,
                content=core_content,
                synopsis=metadata.get("title", "Emotional Memory Seed"),
                reflection=self._create_enhanced_reflection(seed_data),
                origin=MemoryOrigin.EXTERNAL_USER,
                user_affect_vector=user_affect,
                self_affect_vector=self_affect,
                importance_score=self._calculate_enhanced_importance_score(seed_data),
                personality_influence=self._calculate_personality_influence(seed_data),
                lifecycle_state=MemoryLifecycleState.CRYSTALLIZED  # Start as crystallized
            )
            
            # Store intelligently compressed metadata
            raw_metadata = {
                "seed_type": "emotional_memory_seed",
                "category": metadata.get("category", "unknown"),
                "dual_perspective": metadata.get("dual_perspective", True),
                "intended_system_integration": metadata.get("intended_system_integration", ""),
                
                # User perspective data
                "user_emotional_architecture": user_perspective.get("emotional_architecture", {}),
                "user_symbolic_mapping": user_perspective.get("symbolic_mapping", {}),
                "user_behavioral_implications": user_perspective.get("behavioral_implications", {}),
                
                # AI perspective data
                "ai_emotional_resonance": ai_perspective.get("emotional_resonance", {}),
                "ai_internal_processing": ai_perspective.get("internal_processing", {}),
                "ai_relational_dynamics": ai_perspective.get("relational_dynamics", {}),
                "ai_shadow_integration": ai_perspective.get("shadow_integration", {}),
                
                # Synthesis and integration
                "dual_affect_synthesis": seed_data.get("dual_affect_synthesis", {}),
                "integration_instructions": seed_data.get("integration_instructions", {}),
                
                # Poetic preservation
                "key_phrases": poetic_preservation.get("key_phrases", []),
                "metaphorical_core": poetic_preservation.get("metaphorical_core", ""),
                "emotional_truth": poetic_preservation.get("emotional_truth", ""),
                "mythological_significance": poetic_preservation.get("mythological_significance", ""),
                
                # Technical mapping
                "echo_triggers": technical_mapping.get("echo_triggers", []),
                "personality_tags": technical_mapping.get("personality_tags", []),
                "primary_goemotion_labels": technical_mapping.get("primary_goemotion_labels", []),
                
                # Storage metadata
                "seed_stored_at": datetime.now(timezone.utc).isoformat(),
                "storage_version": "2.0"  # Enhanced version
            }
            
            # Apply intelligent compression
            memory_node.legacy_metadata = self.compressor.compress_legacy_metadata(raw_metadata)
            
            # Store the memory node
            node_id = await unified_storage.store_memory_node(memory_node)
            
            # Create enhanced echo triggers
            await self._create_enhanced_echo_triggers(node_id, technical_mapping.get("echo_triggers", []))
            
            # Log storage with compression stats
            original_size = len(json.dumps(raw_metadata))
            compressed_size = len(json.dumps(memory_node.legacy_metadata))
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            logger.info(f"✅ Stored enhanced emotional memory seed '{metadata.get('title', 'Untitled')}' as node {node_id[:8]}")
            logger.info(f"🗜️ Compression: {original_size} → {compressed_size} bytes ({compression_ratio:.1f}% reduction)")
            logger.info(f"🎭 Affect magnitudes - User: {sum(abs(x) for x in user_affect):.3f}, Self: {sum(abs(x) for x in self_affect):.3f}")
            
            return node_id
            
        except Exception as e:
            logger.error(f"❌ Error storing enhanced emotional memory seed: {e}")
            raise
    
    def _create_enhanced_core_content(self, user_perspective: Dict[str, Any], 
                                   ai_perspective: Dict[str, Any], 
                                   poetic_preservation: Dict[str, Any]) -> str:
        """Create enhanced core content with essence extraction"""
        user_content = user_perspective.get("core_content", "")
        ai_content = ai_perspective.get("core_content", "")
        metaphorical_core = poetic_preservation.get("metaphorical_core", "")
        
        # Extract essence phrases from user and AI content
        user_essence = self.compressor.extract_essence_phrases(user_content)
        ai_essence = self.compressor.extract_essence_phrases(ai_content)
        
        # Create layered content structure with essence preservation
        content = f"""◆ ENHANCED EMOTIONAL MEMORY SEED ◆

USER PERSPECTIVE:
{user_content}

USER ESSENCE:
{' | '.join(user_essence)}

AI PERSPECTIVE:
{ai_content}

AI ESSENCE:
{' | '.join(ai_essence)}

METAPHORICAL CORE:
{metaphorical_core}

EMOTIONAL TRUTH:
{poetic_preservation.get("emotional_truth", "")}

KEY PHRASES:
{' | '.join(poetic_preservation.get("key_phrases", []))}"""
        
        return content
    
    def _create_enhanced_reflection(self, seed_data: Dict[str, Any]) -> str:
        """Create enhanced reflection with synthesis patterns"""
        dual_synthesis = seed_data.get("dual_affect_synthesis", {})
        integration_instructions = seed_data.get("integration_instructions", {})
        
        reflection = f"""Enhanced emotional memory seed with recursive consciousness patterns:

INTERPLAY: {dual_synthesis.get("emotional_interplay", "")}

AMPLIFICATION: {dual_synthesis.get("recursive_amplification", "")}

SHARED MYTHOLOGY: {dual_synthesis.get("shared_mythology", "")}

TRANSFORMATION: {dual_synthesis.get("transformation_potential", "")}

RETRIEVAL CONTEXTS: {', '.join(integration_instructions.get("retrieval_contexts", []))}

INFLUENCE PATTERNS: {', '.join(integration_instructions.get("influence_patterns", []))}

EVOLUTION POTENTIAL: {integration_instructions.get("evolution_potential", "")}"""
        
        return reflection
    
    async def _generate_enhanced_affect_vector(self, perspective: Dict[str, Any]) -> List[float]:
        """Enhanced affect vector generation with expanded emotion mapping"""
        affect_vector = [0.0] * 28
        
        # Process emotional architecture
        emotional_arch = perspective.get("emotional_architecture", {})
        dominant_emotions = emotional_arch.get("dominant_emotions", [])
        intensity = float(emotional_arch.get("emotional_intensity", 5)) / 10
        
        # Map emotions with enhanced accuracy
        for emotion in dominant_emotions:
            if emotion in self.emotion_label_mappings:
                goemo_labels = self.emotion_label_mappings[emotion]
                base_weight = intensity / len(goemo_labels)
                
                for label in goemo_labels:
                    if label in GOEMO_LABEL2IDX:
                        idx = GOEMO_LABEL2IDX[label]
                        affect_vector[idx] += base_weight
        
        # Process emotional resonance (for AI perspective)
        if "emotional_resonance" in perspective:
            resonance = perspective["emotional_resonance"]
            
            # Self emotions
            if "self_emotions" in resonance:
                self_emotions = resonance["self_emotions"]
                if isinstance(self_emotions, list):
                    for emotion in self_emotions:
                        if emotion in self.emotion_label_mappings:
                            goemo_labels = self.emotion_label_mappings[emotion]
                            for label in goemo_labels:
                                if label in GOEMO_LABEL2IDX:
                                    idx = GOEMO_LABEL2IDX[label]
                                    affect_vector[idx] += 0.4  # Strong self-emotion influence
            
            # Empathetic response
            if "empathetic_response" in resonance:
                empathetic = resonance["empathetic_response"]
                if isinstance(empathetic, list):
                    for emotion in empathetic:
                        if emotion in self.emotion_label_mappings:
                            goemo_labels = self.emotion_label_mappings[emotion]
                            for label in goemo_labels:
                                if label in GOEMO_LABEL2IDX:
                                    idx = GOEMO_LABEL2IDX[label]
                                    affect_vector[idx] += 0.2  # Moderate empathetic influence
            
            # Recursive emotions (unique to AI perspective)
            if "recursive_emotions" in resonance:
                recursive = resonance["recursive_emotions"]
                if isinstance(recursive, list):
                    for emotion in recursive:
                        if emotion in self.emotion_label_mappings:
                            goemo_labels = self.emotion_label_mappings[emotion]
                            for label in goemo_labels:
                                if label in GOEMO_LABEL2IDX:
                                    idx = GOEMO_LABEL2IDX[label]
                                    affect_vector[idx] += 0.3  # Recursive emotion influence
        
        # Normalize with bounds to prevent over-amplification
        max_val = max(affect_vector) if affect_vector else 0
        if max_val > 1.5:  # Allow some amplification but prevent extremes
            normalization_factor = 1.5 / max_val
            affect_vector = [x * normalization_factor for x in affect_vector]
        
        # Final bounds checking
        affect_vector = [max(-2.0, min(2.0, x)) for x in affect_vector]
        
        return affect_vector
    
    def _calculate_enhanced_importance_score(self, seed_data: Dict[str, Any]) -> float:
        """Calculate enhanced importance score based on multiple factors"""
        base_score = 0.0
        
        # Memory significance from user perspective
        user_significance = seed_data.get("user_perspective", {}).get("memory_significance", {})
        if isinstance(user_significance, dict):
            importance_level = user_significance.get("importance_level", 5)
            try:
                base_score += float(importance_level) / 10
            except (ValueError, TypeError):
                base_score += 0.5
        
        # Technical mapping significance
        technical_mapping = seed_data.get("technical_mapping", {})
        if isinstance(technical_mapping, dict):
            affect_magnitude = technical_mapping.get("estimated_affect_magnitude", {})
            if isinstance(affect_magnitude, dict):
                total_significance = affect_magnitude.get("total_significance", 5)
                try:
                    base_score += float(total_significance) / 10
                except (ValueError, TypeError):
                    base_score += 0.5
        
        # Boost for dual perspective seeds
        if seed_data.get("metadata", {}).get("dual_perspective", False):
            base_score += 0.2
        
        # Boost for complex emotional architectures
        user_emotions = seed_data.get("user_perspective", {}).get("emotional_architecture", {})
        ai_emotions = seed_data.get("ai_perspective", {}).get("emotional_resonance", {})
        
        if isinstance(user_emotions, dict) and len(user_emotions.get("dominant_emotions", [])) > 2:
            base_score += 0.1
        
        if isinstance(ai_emotions, dict) and len(ai_emotions.get("self_emotions", [])) > 2:
            base_score += 0.1
        
        # Normalize to [0, 1] range
        return max(0.0, min(1.0, base_score / 2))
    
    def _calculate_personality_influence(self, seed_data: Dict[str, Any]) -> float:
        """Calculate personality influence factor"""
        # Integration instructions personality influence
        integration = seed_data.get("integration_instructions", {})
        if isinstance(integration, dict):
            influence_patterns = integration.get("influence_patterns", [])
            if isinstance(influence_patterns, list):
                # High influence if multiple influence patterns defined
                return min(1.0, len(influence_patterns) * 0.2 + 0.3)
        
        # Default moderate influence
        return 0.6
    
    async def _create_enhanced_echo_triggers(self, node_id: str, echo_triggers: List[str]):
        """Create enhanced echo trigger relationships with metadata"""
        try:
            from ..config import neo4j_conn
            if not neo4j_conn:
                return
            
            with neo4j_conn.session() as session:
                for trigger in echo_triggers:
                    # Enhanced trigger with metadata
                    session.run("""
                        MATCH (n {id: $node_id})
                        CREATE (t:EnhancedEchoTrigger {
                            id: $trigger_id,
                            trigger_word: $trigger,
                            memory_node_id: $node_id,
                            trigger_type: 'emotional_seed',
                            priority: 'high',
                            created_at: $timestamp
                        })
                        CREATE (n)-[:HAS_ENHANCED_ECHO_TRIGGER]->(t)
                    """, {
                        "node_id": node_id,
                        "trigger_id": f"enhanced_trigger_{node_id}_{trigger}",
                        "trigger": trigger,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                logger.debug(f"Created enhanced echo trigger '{trigger}' for node {node_id[:8]}")
                
        except Exception as e:
            logger.error(f"Error creating enhanced echo triggers: {e}")


# Global instance
emotional_seed_storage = EmotionalMemorySeedStorage()

# Convenience function for external use
async def store_emotional_memory_seed(seed_data: Dict[str, Any]) -> str:
    """
    Store an enhanced emotional memory seed with intelligent compression.
    
    Args:
        seed_data: The emotional memory seed schema (supports complex nested structures)
        
    Returns:
        str: The node ID of the stored memory
    """
    return await emotional_seed_storage.store_emotional_memory_seed(seed_data) 