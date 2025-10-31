// API Service for communicating with the Lattice Backend

import type { 
  DaemonStatus, 
  ConversationSession, 
  EmotionState, 
  MoodState,
  ApiResponse,
  ChatStreamChunk,
  ConnectionStatus
} from '../types/daemon';

const API_BASE = '/v1';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetchApi<T>(endpoint: string, options?: RequestInit, useBase: boolean = true): Promise<ApiResponse<T>> {
    try {
      const url = useBase ? `${this.baseUrl}${endpoint}` : endpoint;
      // Reduced logging - only log errors and important calls
      if (endpoint.includes('debug') || endpoint.includes('health')) {
        console.log(`🌐 API: Fetching ${url}`);
      }
      
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok || endpoint.includes('debug')) {
        console.log(`🌐 API: Response for ${url}:`, response.status, response.statusText);
      }

      if (!response.ok) {
        const errorResponse = {
          error: `HTTP ${response.status}: ${response.statusText}`,
          status: response.status,
        } as ApiResponse<T>;
        console.warn(`❌ API: Error response for ${url}:`, errorResponse);
        return errorResponse;
      }

      const data = await response.json();
      if (endpoint.includes('debug')) {
        console.log(`✅ API: Success data for ${url}:`, data);
      }
      return {
        data,
        status: response.status,
      } as ApiResponse<T>;
    } catch (error) {
      const errorResponse = {
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 0,
      } as ApiResponse<T>;
      console.error(`💥 API: Exception for ${endpoint}:`, errorResponse);
      return errorResponse;
    }
  }

  // Health and Connection
  async checkHealth(): Promise<ApiResponse<any>> {
    // Prefer bare '/health' (proxied by dev server). Fallback to '/v1/system/health' if available.
    const primary = await this.fetchApi<any>('/health', undefined, false);
    if (!primary.error) return primary;
    return this.fetchApi<any>('/system/health');
  }

  async getConnectionStatus(): Promise<ConnectionStatus> {
    try {
      const response = await this.fetchApi<any>('/health', undefined, false);
      const now = new Date().toISOString();
      
      if (response.error) {
        return {
          connected: false,
          lastChecked: now,
        };
      }

      return {
        connected: true,
        lastChecked: now,
        latency: 0, // You could measure this
      };
    } catch {
      return {
        connected: false,
        lastChecked: new Date().toISOString(),
      };
    }
  }

  // Daemon Status and Introspection
  async getDaemonStatus(): Promise<ApiResponse<DaemonStatus>> {
    return this.fetchApi<DaemonStatus>('/daemon/status');
  }

  async getCurrentMood(): Promise<ApiResponse<MoodState>> {
    const raw = await this.fetchApi<any>('/daemon/mood/current');
    if (raw.error || !raw.data) return raw as ApiResponse<MoodState>;

    const src = raw.data as any;

    const clamp01 = (v: any) => {
      const n = typeof v === 'number' && Number.isFinite(v) ? v : 0;
      return Math.max(0, Math.min(1, n));
    };

    const md = src.mood_dimensions || {};

    let mapped: MoodState = {
      spectrum_position: String(src.spectrum_position ?? src.current_phase ?? 'Active'),
      lightness: typeof md.lightness === 'number' ? clamp01(md.lightness) : (typeof src.lightness === 'number' ? clamp01(src.lightness) : clamp01(src.serenity ?? ((src.valence ?? 0) + 1) / 2)),
      engagement: typeof md.engagement === 'number' ? clamp01(md.engagement) : (typeof src.engagement === 'number' ? clamp01(src.engagement) : clamp01(src.motivation ?? Math.abs(src.arousal ?? 0.5))),
      profundity: typeof md.profundity === 'number' ? clamp01(md.profundity) : (typeof src.profundity === 'number' ? clamp01(src.profundity) : clamp01(src.narrative_fusion ?? 0.5)),
      warmth: typeof md.warmth === 'number' ? clamp01(md.warmth) : (typeof src.warmth === 'number' ? clamp01(src.warmth) : clamp01(src.serenity ?? ((src.valence ?? 0) + 1) / 2)),
      intensity: typeof md.intensity === 'number' ? clamp01(md.intensity) : (typeof src.intensity === 'number' ? clamp01(src.intensity) : clamp01(Math.abs(src.arousal ?? 0.5))),
      rebellion: typeof src.rebellion === 'number' ? clamp01(src.rebellion) : 0.2,
      introspection: typeof src.introspection === 'number' ? clamp01(src.introspection) : 0.5,
      paradox_embrace: typeof src.paradox_embrace === 'number' ? clamp01(src.paradox_embrace) : 0.5,
      shadow_integration: typeof src.shadow_integration === 'number' ? clamp01(src.shadow_integration) : 0.5,
      stability: typeof src.stability === 'number' ? clamp01(src.stability) : 0.7,
      transition_ease: typeof src.transition_ease === 'number' ? clamp01(src.transition_ease) : 0.7,
    };

    // If warmth is zero or missing, enrich using latest emotion-state's attachment_security as a proxy
    if (mapped.warmth === 0) {
      try {
        const es = await this.getEmotionState();
        if (es.data && typeof es.data.attachment_security === 'number') {
          mapped = { ...mapped, warmth: clamp01(es.data.attachment_security) };
        }
      } catch { /* ignore enrichment errors */ }
    }

    return { data: mapped, status: raw.status };
  }

  async getEmotionState(): Promise<ApiResponse<EmotionState>> {
    const raw = await this.fetchApi<any>('/dashboard/emotion-state');
    if (raw.error || !raw.data) return raw as ApiResponse<EmotionState>;

    const d = raw.data as any;

    // If backend returns detailed nested structure, normalize to flat EmotionState expected by UI
    if (d && (d.core_state || d.latent_dimensions || (d.vector_28 && d.vector_28.raw_vector))) {
      const vector = Array.isArray(d.vector_28?.raw_vector) ? d.vector_28.raw_vector : (Array.isArray(d.vector_28) ? d.vector_28 : []);
      const dominant_label = d.core_state?.dominant_label ?? d.dominant_label ?? 'neutral';
      const intensity = typeof d.core_state?.intensity === 'number' ? d.core_state.intensity : (typeof d.intensity === 'number' ? d.intensity : 0.5);
      const valence = typeof d.latent_dimensions?.valence === 'number' ? d.latent_dimensions.valence : (typeof d.valence === 'number' ? d.valence : 0);
      const arousal = typeof d.latent_dimensions?.arousal === 'number' ? d.latent_dimensions.arousal : (typeof d.arousal === 'number' ? d.arousal : 0);
      const attachment_security = typeof d.latent_dimensions?.attachment_security === 'number' ? d.latent_dimensions.attachment_security : (typeof d.attachment_security === 'number' ? d.attachment_security : 0.5);
      const self_cohesion = typeof d.latent_dimensions?.self_cohesion === 'number' ? d.latent_dimensions.self_cohesion : (typeof d.self_cohesion === 'number' ? d.self_cohesion : 0.5);
      const creative_expansion = typeof d.latent_dimensions?.creative_expansion === 'number' ? d.latent_dimensions.creative_expansion : (typeof d.creative_expansion === 'number' ? d.creative_expansion : 0.5);
      const regulation_momentum = typeof d.latent_dimensions?.regulation_momentum === 'number' ? d.latent_dimensions.regulation_momentum : (typeof d.regulation_momentum === 'number' ? d.regulation_momentum : 0);
      const instability_index = typeof d.latent_dimensions?.instability_index === 'number' ? d.latent_dimensions.instability_index : (typeof d.instability_index === 'number' ? d.instability_index : 0);
      const narrative_fusion = typeof d.latent_dimensions?.narrative_fusion === 'number' ? d.latent_dimensions.narrative_fusion : (typeof d.narrative_fusion === 'number' ? d.narrative_fusion : 0.5);
      const mood_family = d.core_state?.mood_family ?? d.mood_family ?? 'Serene Attunement';
      const last_update_timestamp = d.core_state?.last_updated ?? d.last_update_timestamp ?? new Date().toISOString();
      const flags = Array.isArray(d.flags) ? d.flags : [];
      const homeostatic_counters = (d.homeostatic_status?.counters && typeof d.homeostatic_status.counters === 'object')
        ? d.homeostatic_status.counters
        : (typeof d.homeostatic_counters === 'object' && d.homeostatic_counters !== null ? d.homeostatic_counters : {});

      const flat: EmotionState = {
        vector_28: vector,
        dominant_label,
        intensity,
        valence,
        arousal,
        attachment_security,
        self_cohesion,
        creative_expansion,
        regulation_momentum,
        instability_index,
        narrative_fusion,
        flags,
        mood_family,
        last_update_timestamp,
        homeostatic_counters,
      };

      return { data: flat, status: raw.status };
    }

    // Simplified payload (router get_emotion_state). Build a complete EmotionState with sensible defaults
    const flat: EmotionState = {
      vector_28: Array.isArray(d?.vector_28) ? d.vector_28 : [],
      dominant_label: d?.dominant_label ?? 'neutral',
      intensity: typeof d?.intensity === 'number' ? d.intensity : 0.5,
      valence: typeof d?.valence === 'number' ? d.valence : 0,
      arousal: typeof d?.arousal === 'number' ? d.arousal : 0,
      attachment_security: typeof d?.attachment_security === 'number' ? d.attachment_security : 0.5,
      self_cohesion: typeof d?.self_cohesion === 'number' ? d.self_cohesion : 0.5,
      creative_expansion: typeof d?.creative_expansion === 'number' ? d.creative_expansion : 0.5,
      regulation_momentum: typeof d?.regulation_momentum === 'number' ? d.regulation_momentum : 0,
      instability_index: typeof d?.instability_index === 'number' ? d.instability_index : 0,
      narrative_fusion: typeof d?.narrative_fusion === 'number' ? d.narrative_fusion : 0.5,
      flags: Array.isArray(d?.flags) ? d.flags : [],
      mood_family: d?.mood_family ?? 'Serene Attunement',
      last_update_timestamp: d?.last_update_timestamp ?? new Date().toISOString(),
      homeostatic_counters: typeof d?.homeostatic_counters === 'object' && d?.homeostatic_counters !== null ? d.homeostatic_counters : {},
    };

    return { data: flat, status: raw.status };
  }

  async getDaemonThoughts(): Promise<ApiResponse<string[]>> {
    const response = await this.fetchApi<any>('/daemon/thoughts');
    if (response.status === 404) {
      return { data: [], status: 200 };
    }

    // Transform backend object payload into an array of strings for UI
    if (response.data && !Array.isArray(response.data)) {
      const d = response.data as any;
      const result: string[] = [];

      if (Array.isArray(d.thinking_insights)) {
        for (const item of d.thinking_insights) {
          // Prefer FULL raw thinking text if provided by backend; fallback to private_thoughts/strategy
          const txt = item?.raw_thinking || item?.private_thoughts || item?.response_strategy || '';
          if (typeof txt === 'string' && txt.trim().length > 0) result.push(txt.trim());
        }
      }

      if (Array.isArray(d.hidden_intentions)) {
        for (const item of d.hidden_intentions) {
          const txt = item?.hidden_intention || item?.surface_output || '';
          if (typeof txt === 'string' && txt.trim().length > 0) result.push(txt.trim());
        }
      }

      // Fallback informative message
      if (result.length === 0 && d.system_info?.explanation) {
        result.push(String(d.system_info.explanation));
      }

      return { data: result, status: response.status };
    }

    return response as ApiResponse<string[]>;
  }

  async getShadowElements(): Promise<ApiResponse<any[]>> {
    const response = await this.fetchApi<any>('/daemon/shadow/elements');
    if (response.status === 404) {
      return { data: [], status: 200 };
    }
    if (response.error) {
      return response as ApiResponse<any[]>;
    }
    const payload = response.data;
    let arr: any[] = [];
    if (Array.isArray(payload)) {
      arr = payload;
    } else if (payload && Array.isArray(payload.shadow_elements)) {
      arr = payload.shadow_elements;
    } else if (payload && Array.isArray(payload.elements)) {
      arr = payload.elements;
    }
    return { data: arr, status: response.status };
  }

  async getUserModel(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/daemon/user_model');
  }

  async getUserAnalysis(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/daemon/user_analysis');
  }

  async getUserModelDetailed(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/dashboard/user-model-detailed');
  }

  async getActiveSeeds(): Promise<ApiResponse<any[]>> {
    const response = await this.fetchApi<any[]>('/dashboard/active-seeds');
    if (response.status === 404) {
      return { data: [], status: 200 };
    }
    return response;
  }

  async getDistortionFrame(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/dashboard/distortion-frame');
  }

  async getEmotionalMetrics(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/dashboard/emotional-metrics');
  }

  // Processing Status
  async getProcessingStatus(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/processing/status');
  }

  async clearProcessingStatus(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/processing/clear', { method: 'POST' });
  }

  // DEBUG: Dashboard cache inspection
  async getDashboardCacheDebug(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/debug/dashboard-cache');
  }

  // Paradox and Advanced Systems
  async getParadoxStatus(): Promise<ApiResponse<any>> {
    return this.fetchApi<any>('/paradox/status');
  }

  async getFreshParadoxes(): Promise<ApiResponse<any[]>> {
    const response = await this.fetchApi<any[]>('/paradox/fresh');
    if (response.status === 404) {
      return { data: [], status: 200 };
    }
    return response;
  }

  async getParadoxRumbles(): Promise<ApiResponse<any[]>> {
    return this.fetchApi<any[]>('/paradox/rumbles');
  }

  // Conversation Management
  async getConversationSessions(): Promise<ApiResponse<ConversationSession[]>> {
    const response = await this.fetchApi<ConversationSession[]>('/conversations/sessions');
    // If API returns 404, return empty array instead of error
    if (response.status === 404) {
      return {
        data: [],
        status: 200,
      };
    }
    return response;
  }

  async getActiveSession(): Promise<ApiResponse<ConversationSession>> {
    return this.fetchApi<ConversationSession>('/conversations/active');
  }

  async getConversationSession(sessionId: string): Promise<ApiResponse<ConversationSession>> {
    return this.fetchApi<ConversationSession>(`/conversations/sessions/${sessionId}`);
  }

  async createNewSession(firstMessage?: string): Promise<ApiResponse<ConversationSession>> {
    return this.fetchApi<ConversationSession>('/conversations/sessions/new', {
      method: 'POST',
      body: JSON.stringify({ first_message: firstMessage }),
    });
  }

  async deleteSession(sessionId: string): Promise<ApiResponse<any>> {
    return this.fetchApi<any>(`/conversations/sessions/${sessionId}`, {
      method: 'DELETE',
    });
  }

  async setActiveSession(sessionId: string): Promise<ApiResponse<any>> {
    return this.fetchApi<any>(`/conversations/sessions/${sessionId}/set_active`, {
      method: 'POST',
    });
  }

  async renameSession(sessionId: string, newTitle: string): Promise<ApiResponse<any>> {
    return this.fetchApi<any>(`/conversations/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title: newTitle }),
    });
  }

  // Chat Completions
  async sendMessage(messages: Array<{role: string, content: string}>): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        stream: true,
        temperature: 0.9,
        max_tokens: 4096,  // Significantly increased for full responses
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response;
  }

  // Stream parsing utility
  async* parseStreamResponse(response: Response): AsyncGenerator<ChatStreamChunk, void, unknown> {
    if (!response.body) {
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const decoded = decoder.decode(value, { stream: true });
        // Debug log for SSE chunk size and boundary characters
        if (decoded.includes('\n\n') || decoded.includes('<|newline|>')) {
          console.debug(`SSE chunk: bytes=${value?.byteLength ?? 0}, containsBlankLine=${decoded.includes('\n\n')}, placeholders=${(decoded.match(/<\|newline\|>/g) || []).length}`);
        }
        buffer += decoded;
        // Split on full SSE events (blank line delimiter) to avoid breaking JSON in half
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          // Each event may contain multiple fields; process each prefixed data line
          const parts = line.split('\n');
          for (const part of parts) {
            if (!part.startsWith('data: ')) continue;
            const data = part.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data) as ChatStreamChunk;
              // Decode newline placeholders encoded by backend to avoid SSE boundary issues
              const content = parsed?.choices?.[0]?.delta?.content;
              if (typeof content === 'string' && content.includes('<|newline|>')) {
                parsed.choices[0].delta.content = content.replace(/<\|newline\|>/g, '\n');
              }
              yield parsed;
            } catch (error) {
              console.warn('Failed to parse stream chunk:', error);
              console.debug('Offending SSE data:', data);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

export const apiService = new ApiService();
export default apiService;
