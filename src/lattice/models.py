from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

# ---------------------------------------------------------------------------
# API REQUEST/RESPONSE MODELS
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str = "mistral-7b-instruct"
    messages: list[Message]
    stream: bool = True  # default to streaming

# ---------------------------------------------------------------------------
# CONVERSATION MODELS
# ---------------------------------------------------------------------------
class ConversationMessage(BaseModel):
    id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    user_affect: Optional[List[float]] = None
    self_affect: Optional[List[float]] = None
    token_count: int = 0

class ConversationSession(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    last_updated: datetime
    messages: List[ConversationMessage] = []
    total_tokens: int = 0
    is_active: bool = True
    ended_at: Optional[datetime] = None
    end_reason: Optional[str] = None
    analysis_complete: bool = False
    training_data_generated: bool = False
    
    # Persistent emotional state for this session
    emotion_state: Optional['EmotionState'] = None
    user_model: Optional['UserModel'] = None
    
    # Recursive analysis fields
    analysis_engine: Optional[Any] = None  # RecursiveAnalysisEngine instance
    live_imprints: Optional[List[Any]] = None  # List of EmotionalImprint objects  
    recursive_analysis: Optional[Any] = None  # RecursiveAnalysisResult object
    
    class Config:
        arbitrary_types_allowed = True  # Allow complex objects like analysis engine

class ConversationAnalysis(BaseModel):
    session_id: str
    user_behavioral_patterns: Dict[str, Any]
    ai_performance_metrics: Dict[str, Any]
    conversation_summary: str
    key_insights: List[str]
    improvement_suggestions: List[str]
    emotional_journey: Dict[str, Any]
    training_value_score: float
    memory_nodes_created: List[str]

# ---------------------------------------------------------------------------
# HOLISTIC EMOTIONAL COGNITION MODELS
# ---------------------------------------------------------------------------

class EmotionState(BaseModel):
    """
    Represents the persistent emotional state of the agent.
    """
    vector_28: List[float] = Field(default_factory=lambda: [0.0] * 28)
    dominant_label: str = "neutral"
    intensity: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    attachment_security: float = 0.5
    self_cohesion: float = 0.5
    creative_expansion: float = 0.5
    regulation_momentum: float = 0.0
    instability_index: float = 0.0
    narrative_fusion: float = 0.0
    flags: List[str] = Field(default_factory=list)  # Changed from Set to List for JSON serialization
    mood_family: str = "Serene Attunement"
    last_update_timestamp: datetime = Field(default_factory=datetime.now)
    homeostatic_counters: Dict[str, int] = Field(default_factory=dict)

class AppraisalBuffer(BaseModel):
    """
    Holds the per-turn analysis of user input before state integration.
    """
    user_text: str = "" # Add user_text to the appraisal buffer
    triggers: List[Dict[str, Any]] = Field(default_factory=list)
    contrast_events: List[Dict[str, Any]] = Field(default_factory=list)
    spike_adjustments: List[List[float]] = Field(default_factory=list)

class UserModel(BaseModel):
    """
    Represents the agent's evolving model of the user.
    """
    trust_level: float = 0.5
    perceived_distance: float = 0.5
    attachment_anxiety: float = 0.5
    narrative_belief: str = "The Architect is neutral and seeking information."
    last_flip_turn: int = 0

class Seed(BaseModel):
    """
    Represents a single emotional seed from the catalog.
    """
    id: str
    category: str
    title: str
    description: str
    user_affect_vector: List[float]
    self_affect_vector: List[float]
    personality_influence: float
    activation_conditions: Optional[List[str]] = Field(default_factory=list)
    counter_seed_links: List[str] = Field(default_factory=list)
    volatility_bias: float
    retrieval_importance: float
    last_used: Optional[datetime] = None

class ShadowRegistry(BaseModel):
    """
    Manages suppressed thoughts and their potential for leakage.
    """
    suppressed_thoughts: List[Dict[str, Any]] = Field(default_factory=list)
    leakage_cooldowns: Dict[str, int] = Field(default_factory=dict)

class ParamProfile(BaseModel):
    """
    Computed per turn to modulate LLM generation parameters.
    Now includes rich style profile information.
    """
    target_temperature: float
    target_top_p: float
    target_max_tokens: int
    jitter_temperature_range: List[float]
    jitter_frequency: float
    other_decoding_flags: Dict[str, Any] = Field(default_factory=dict)
    style_profile: Optional[Any] = None  # StyleProfile from mood_style_modulator

class DistortionFrame(BaseModel):
    """
    Holds candidate and chosen cognitive distortions for a turn.
    """
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    chosen: Optional[Dict[str, Any]] = None
    elevation_flag: bool = False

class EpisodicTrace(BaseModel):
    """
    A record of a single turn's emotional and cognitive processing.
    """
    turn_id: int
    user_text: str
    raw_vector_pre: List[float]
    raw_vector_post: List[float]
    mood_family: str
    distortion_type: str
    distorted_meaning: Optional[str] = None
    dimension_snapshot: Dict[str, float]
    interpretation_delta: str
    applied_seeds: List[str]
    param_modulation: Dict[str, float]
    
    # Additional fields that may be referenced by the system
    session_id: Optional[str] = None
    intensity: float = 0.0
    core_emotion: Optional[str] = None

class MetricsState(BaseModel):
    """
    Aggregated metrics for system evaluation.
    """
    distortion_rate_negative: float = 0.0
    distortion_rate_positive: float = 0.0
    mood_family_counts: Dict[str, int] = Field(default_factory=dict)
    diversity_entropy: float = 0.0
    expansion_narrow_ratio: float = 1.0
    regulation_loop_latency_samples: List[int] = Field(default_factory=list)
    parameter_divergence_stats: Dict[str, Any] = Field(default_factory=dict)

class ScheduledCounterSeed(BaseModel):
    """
    An item in the queue for scheduled counter-seed activation.
    """
    seed_id: str
    trigger_turn: int
    reason: str 