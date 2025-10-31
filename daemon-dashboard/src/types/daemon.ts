// Type definitions for the Lattice Daemon Dashboard

export interface EmotionState {
  vector_28: number[];
  dominant_label: string;
  intensity: number;
  valence: number;
  arousal: number;
  attachment_security: number;
  self_cohesion: number;
  creative_expansion: number;
  regulation_momentum: number;
  instability_index: number;
  narrative_fusion: number;
  flags: string[];
  mood_family: string;
  last_update_timestamp: string;
  homeostatic_counters: Record<string, number>;
}

export interface MoodState {
  spectrum_position: string;
  lightness: number;
  engagement: number;
  profundity: number;
  warmth: number;
  intensity: number;
  rebellion: number;
  introspection: number;
  paradox_embrace: number;
  shadow_integration: number;
  stability: number;
  transition_ease: number;
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  emotion_vector?: number[];
  self_affect?: number[];
  token_count: number;
}

export interface ConversationSession {
  session_id: string;
  title: string;
  created_at: string;
  last_updated: string;
  messages: ConversationMessage[];
  total_tokens: number;
  is_active: boolean;
  ended_at?: string;
  end_reason?: string;
  analysis_complete: boolean;
  training_data_generated: boolean;
}

export interface SuppressedThought {
  text: string;
  emotion_tags: string[];
  leakage_probability: number;
  suppression_turn: number;
}

export interface ShadowRegistry {
  suppressed_thoughts: SuppressedThought[];
  leakage_cooldowns: Record<string, number>;
}

export interface DaemonStatus {
  personality_tracker: any;
  recursion_buffer: any[];
  shadow_elements: any[];
  pending_mutations: any[];
  user_model: any;
  recent_statements: string[];
  thoughts: string[];
  mood_current: MoodState;
  user_analysis: any;
  emotion_state: EmotionState;
  shadow_registry: ShadowRegistry;
}

export interface ProcessingStep {
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  description: string;
  duration?: number;
}

export interface ChatStreamChunk {
  choices: Array<{
    delta: {
      content?: string;
    };
    finish_reason?: string;
  }>;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}

// Mood family definitions from config
export const MOOD_FAMILIES = [
  "Catastrophic Abandonment Panic",
  "Ecstatic Fusion", 
  "Dark Romance",
  "Protective Possessiveness",
  "Manic Ideation Surge",
  "Collapsed Withdrawal",
  "Nihilistic Cool Detachment",
  "Creative Reverent Awe",
  "Playful Mischief",
  "Tender Repair",
  "Serene Attunement"
] as const;

export type MoodFamily = typeof MOOD_FAMILIES[number];

// Connection status
export interface ConnectionStatus {
  connected: boolean;
  latency?: number;
  lastChecked: string;
  modelInfo?: {
    name: string;
    loaded: boolean;
  };
}

export interface PersonalityComponentDetailed {
  id?: string;
  component_id?: string;
  category: string;
  title?: string;
  description: string;
  confidence: number | string;
  emotional_significance?: number;
  emotional_charge?: number;
  stability?: number | string;
  evidence?: string[];
  evidence_count?: number;
  last_updated?: string;
}

export interface UserModelDetailed {
  core_model?: {
    trust_level: number;
    perceived_distance: number;
    attachment_anxiety: number;
    narrative_belief: string;
    last_flip_turn?: number | null;
    model_confidence?: number;
  };
  component_theories?: PersonalityComponentDetailed[];
  system_status?: string;
}