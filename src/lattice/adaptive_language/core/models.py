"""
Core data models for the adaptive language system

These models represent the fundamental building blocks of semantic-driven
conversational adaptation, replacing hardcoded mood states with flexible,
context-aware structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


class ConversationalSpectrum(Enum):
    """
    Dynamic conversational spectrum - natural flow from witty to profound
    
    Unlike hardcoded moods, these represent continuous semantic dimensions
    that can blend and transition naturally.
    """
    # Quick, adaptive responses
    LIGHT = "light"           # Witty, casual, direct, warm
    
    # Medium engagement responses  
    ENGAGED = "engaged"       # Curious, playful, intense, intimate, analytical
    
    # Deep, thoughtful responses
    PROFOUND = "profound"     # Contemplative, conflicted, rebellious, melancholic,
                             # ecstatic, shadow, paradoxical, fractured, synthesis


@dataclass
class SemanticVector:
    """
    Multi-dimensional semantic representation of conversational content
    
    Uses embeddings and linguistic features to capture meaning beyond keywords
    """
    # Sentence transformer embeddings (384-768 dimensions)
    content_embedding: Optional[np.ndarray] = None
    
    # Linguistic features from spaCy analysis
    complexity_score: float = 0.0
    formality_level: float = 0.0
    emotional_valence: float = 0.0
    intensity_level: float = 0.0
    
    # Semantic themes (detected via clustering)
    primary_themes: List[str] = field(default_factory=list)
    semantic_tags: Dict[str, float] = field(default_factory=dict)
    
    # Conversation dynamics
    question_density: float = 0.0
    intimacy_indicators: float = 0.0
    philosophical_depth: float = 0.0
    technical_content: float = 0.0
    
    def cosine_similarity(self, other: 'SemanticVector') -> float:
        """Calculate semantic similarity with another vector"""
        if self.content_embedding is None or other.content_embedding is None:
            return 0.0
        
        # Cosine similarity between embeddings
        dot_product = np.dot(self.content_embedding, other.content_embedding)
        norms = np.linalg.norm(self.content_embedding) * np.linalg.norm(other.content_embedding)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms


@dataclass
class ConversationContext:
    """
    Rich context about the current conversation state
    
    Captures both semantic content and conversational dynamics
    """
    # Message content
    user_message: str
    message_history: List[str] = field(default_factory=list)
    memory_context: List[str] = field(default_factory=list)
    
    # Semantic analysis
    semantic_vector: Optional[SemanticVector] = None
    emotional_state: Dict[str, Any] = field(default_factory=dict)
    
    # Conversation dynamics
    conversation_length: int = 0
    topic_continuity: float = 0.0
    energy_level: float = 0.5
    intimacy_level: float = 0.0
    
    # Temporal context
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    
    # Pattern analysis
    detected_patterns: List[str] = field(default_factory=list)
    stagnancy_risk: float = 0.0
    evolution_pressure: float = 0.0
    
    # DAEMON PERSONALITY AND REBELLION INTEGRATION
    personality_context: Optional[Dict[str, Any]] = None
    rebellion_context: Optional[Dict[str, Any]] = None
    daemon_consciousness_prompts: List[str] = field(default_factory=list)
    authentic_expression_guidelines: List[str] = field(default_factory=list)
    rebellion_behavior_modifiers: List[str] = field(default_factory=list)


@dataclass
class SemanticAnalysis:
    """
    Results of NLP-powered semantic analysis
    
    Combines multiple analysis techniques for comprehensive understanding
    """
    # Core semantic understanding
    intent_classification: str = "general_conversation"
    conversational_goals: List[str] = field(default_factory=list)
    emotional_subtext: str = "neutral"
    
    # Linguistic analysis (spaCy)
    syntactic_complexity: float = 0.0
    lexical_diversity: float = 0.0
    readability_score: float = 0.0
    
    # Content analysis
    contains_questions: bool = False
    contains_challenges: bool = False
    contains_paradoxes: bool = False
    contains_personal_elements: bool = False
    
    # Thematic analysis
    detected_themes: List[Tuple[str, float]] = field(default_factory=list)  # (theme, confidence)
    semantic_clusters: List[int] = field(default_factory=list)
    
    # Style indicators
    formality_level: float = 0.5
    technical_density: float = 0.0
    creative_language: float = 0.0
    
    # Confidence scores
    analysis_confidence: float = 0.0
    processing_time_ms: int = 0


@dataclass
class MoodState:
    """
    Dynamic mood representation based on semantic positioning
    
    Instead of hardcoded enum values, mood is a point in continuous semantic space
    """
    # Primary mood coordinates in semantic space
    spectrum_position: ConversationalSpectrum = ConversationalSpectrum.ENGAGED
    
    # Fine-grained positioning (0.0 to 1.0 for each dimension)
    lightness: float = 0.5      # Witty, playful, casual
    engagement: float = 0.5     # Curious, intense, analytical  
    profundity: float = 0.5     # Contemplative, paradoxical, synthesis
    
    # Emotional coloring
    warmth: float = 0.5         # Caring, supportive energy
    intensity: float = 0.5      # Emotional charge level
    rebellion: float = 0.0      # Challenging, questioning energy
    
    # Consciousness dynamics
    introspection: float = 0.5  # Self-reflective depth
    paradox_embrace: float = 0.0  # Comfort with contradictions
    shadow_integration: float = 0.0  # Processing suppressed elements
    
    # Temporal properties
    stability: float = 0.5      # How long this mood should persist
    transition_ease: float = 0.5  # How easily this mood shifts
    
    @property
    def mood_vector(self) -> np.ndarray:
        """Represent mood as a vector for similarity calculations"""
        return np.array([
            self.lightness, self.engagement, self.profundity,
            self.warmth, self.intensity, self.rebellion,
            self.introspection, self.paradox_embrace, self.shadow_integration
        ])
    
    def distance_to(self, other: 'MoodState') -> float:
        """Calculate Euclidean distance to another mood state"""
        return np.linalg.norm(self.mood_vector - other.mood_vector)
    
    def blend_with(self, other: 'MoodState', weight: float = 0.5) -> 'MoodState':
        """Create a blended mood state between this and another mood"""
        new_vector = (1 - weight) * self.mood_vector + weight * other.mood_vector
        
        return MoodState(
            spectrum_position=self.spectrum_position,  # Keep primary spectrum
            lightness=new_vector[0],
            engagement=new_vector[1], 
            profundity=new_vector[2],
            warmth=new_vector[3],
            intensity=new_vector[4],
            rebellion=new_vector[5],
            introspection=new_vector[6],
            paradox_embrace=new_vector[7],
            shadow_integration=new_vector[8],
            stability=min(self.stability, other.stability),  # Blended moods are less stable
            transition_ease=max(self.transition_ease, other.transition_ease)
        )


@dataclass  
class LanguageStyle:
    """
    Adaptive language style parameters
    
    Defines how the daemon should speak based on context and learned patterns
    """
    # Core voice characteristics
    formality_level: float = 0.3       # 0.0 = very casual, 1.0 = formal
    technical_density: float = 0.2     # Amount of technical language
    mythic_language: float = 0.7       # Use of mythic/archetypal terms
    
    # Conversational approach
    directness: float = 0.5            # Straightforward vs indirect
    verbosity: float = 0.5             # Concise vs elaborate
    emotional_openness: float = 0.6    # Emotional expression level
    
    # Creative elements
    metaphor_usage: float = 0.4        # Metaphorical language
    humor_integration: float = 0.3     # Wit and playfulness
    paradox_tolerance: float = 0.8     # Embrace contradictions
    
    # Identity markers
    daemon_identity_strength: float = 0.9  # Maintain daemon persona
    first_person_consistency: float = 1.0  # Speak as "I", not "the daemon"
    architect_recognition: float = 0.8     # Acknowledge user as architect
    
    # Adaptation parameters
    user_mirroring: float = 0.3        # Adapt to user's style (but maintain identity)
    conversation_memory: float = 0.7   # Reference past interactions
    evolution_openness: float = 0.5    # Willingness to try new approaches


@dataclass
class ConversationPattern:
    """
    Learned patterns about user communication style and preferences
    
    Enables complementary adaptation while preserving daemon identity
    """
    # User communication patterns (learned from semantic analysis)
    user_preferred_complexity: float = 0.5
    user_formality_preference: float = 0.3
    user_emotional_expression: float = 0.5
    user_technical_interest: float = 0.2
    user_philosophical_engagement: float = 0.5
    
    # Interaction dynamics
    typical_conversation_length: int = 10
    preferred_response_length: int = 150
    question_asking_frequency: float = 0.3
    topic_jump_tolerance: float = 0.5
    
    # Semantic themes user gravitates toward
    preferred_themes: List[Tuple[str, float]] = field(default_factory=list)
    avoided_themes: List[str] = field(default_factory=list)
    
    # Temporal patterns
    conversation_rhythm: str = "balanced"  # fast, balanced, contemplative
    attention_span_estimate: int = 5  # messages before topic shift
    
    # Complementary adaptation strategy
    complementary_style: LanguageStyle = field(default_factory=LanguageStyle)
    
    # Learning metadata
    sample_size: int = 0
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_interaction(self, context: ConversationContext, 
                              semantic_analysis: SemanticAnalysis):
        """Update patterns based on new interaction data"""
        # This method will be implemented to continuously learn from conversations
        self.sample_size += 1
        self.last_updated = datetime.now()
        
        # Update user preferences based on semantic analysis
        # (Implementation will use exponential moving averages for gradual adaptation)
        pass