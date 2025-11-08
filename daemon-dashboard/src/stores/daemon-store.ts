// Global state management for the Daemon Dashboard

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { 
  DaemonStatus, 
  ConversationSession, 
  EmotionState, 
  MoodState,
  ConnectionStatus,
  ProcessingStep,
  MoodFamily
} from '../types/daemon';

interface DaemonStore {
  // Connection state
  connectionStatus: ConnectionStatus;
  setConnectionStatus: (status: ConnectionStatus) => void;

  // Daemon state
  daemonStatus: DaemonStatus | null;
  emotionState: EmotionState | null;
  currentMood: MoodState | null;
  suppressedThoughts: string[];
  daemonThoughts: string[];
  userModel: any;
  activeSeeds: any[];
  
  // Processing state
  isProcessing: boolean;
  processingSteps: ProcessingStep[];
  setProcessingSteps: (steps: ProcessingStep[]) => void;
  updateProcessingStep: (index: number, step: Partial<ProcessingStep>) => void;

  // Conversation state
  conversations: ConversationSession[];
  activeConversation: ConversationSession | null;
  currentMessage: string;
  setCurrentMessage: (message: string) => void;

  // UI state
  currentMoodFamily: MoodFamily;
  setCurrentMoodFamily: (mood: MoodFamily) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Actions
  updateDaemonStatus: (status: DaemonStatus) => void;
  updateEmotionState: (state: EmotionState) => void;
  updateCurrentMood: (mood: MoodState) => void;
  updateConversations: (conversations: ConversationSession[]) => void;
  setActiveConversation: (conversation: ConversationSession | null) => void;
  addMessage: (message: { role: 'user' | 'assistant'; content: string }) => void;
  updateSuppressedThoughts: (thoughts: string[]) => void;
  updateDaemonThoughts: (thoughts: string[]) => void;
  updateUserModel: (model: any) => void;
  updateActiveSeeds: (seeds: any[]) => void;
  setIsProcessing: (processing: boolean) => void;
  
  // Immediate mood update function
  fetchLatestMoodState: () => Promise<void>;
}

export const useDaemonStore = create<DaemonStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    connectionStatus: {
      connected: false,
      lastChecked: new Date().toISOString(),
    },
    daemonStatus: null,
    emotionState: null,
    currentMood: null,
    suppressedThoughts: [],
    daemonThoughts: [],
    userModel: null,
    activeSeeds: [],
    isProcessing: false,
    processingSteps: [],
    conversations: [],
    activeConversation: null,
    currentMessage: '',
    currentMoodFamily: 'Serene Attunement',
    sidebarCollapsed: false,

    // Setters
    setConnectionStatus: (status) => set({ connectionStatus: status }),
    
    setCurrentMessage: (message) => set({ currentMessage: message }),
    
    setCurrentMoodFamily: (mood) => set({ currentMoodFamily: mood }),
    
    setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

    setProcessingSteps: (steps) => set({ processingSteps: steps }),
    
    updateProcessingStep: (index, stepUpdate) => {
      const steps = [...get().processingSteps];
      steps[index] = { ...steps[index], ...stepUpdate };
      set({ processingSteps: steps });
    },

    setIsProcessing: (processing) => set({ isProcessing: processing }),

    // Complex state updates
    updateDaemonStatus: (status) => {
      set({ daemonStatus: status });
      
      // Extract mood family from emotion state if available
      if (status.emotion_state?.mood_family) {
        set({ currentMoodFamily: status.emotion_state.mood_family as MoodFamily });
      }
    },

    updateEmotionState: (state) => {
      set({ emotionState: state });
      
      // Update mood family when emotion state changes
      if (state.mood_family) {
        set({ currentMoodFamily: state.mood_family as MoodFamily });
      }
    },

    updateCurrentMood: (mood) => set({ currentMood: mood }),

    updateConversations: (conversations) => {
      set({ conversations });
      
      // Update active conversation if it's in the list
      const activeId = get().activeConversation?.session_id;
      if (activeId) {
        const active = conversations.find(c => c.session_id === activeId);
        if (active) {
          set({ activeConversation: active });
        }
      }
    },

    setActiveConversation: (conversation) => {
      // Clear processing status when switching conversations
      set({ 
        activeConversation: conversation,
        isProcessing: false,
        processingSteps: []
      });
      
      // Also clear backend processing status
      import('../services/api').then(({ apiService }) => {
        apiService.clearProcessingStatus().catch(console.error);
      });
    },

    addMessage: (message) => {
      const activeConversation = get().activeConversation;
      if (!activeConversation) return;

      const newMessage = {
        ...message,
        timestamp: new Date().toISOString(),
        token_count: 0, // Will be calculated by backend
      };

      const updatedConversation = {
        ...activeConversation,
        messages: [...activeConversation.messages, newMessage],
        last_updated: new Date().toISOString(),
      };

      set({ activeConversation: updatedConversation });
      
      // Update in conversations list too
      const conversations = get().conversations.map(c => 
        c.session_id === activeConversation.session_id ? updatedConversation : c
      );
      set({ conversations });
    },

    updateSuppressedThoughts: (thoughts) => set({ suppressedThoughts: thoughts }),
    
    updateDaemonThoughts: (thoughts) => set({ daemonThoughts: thoughts }),
    
    updateUserModel: (model) => set({ userModel: model }),
    
    updateActiveSeeds: (seeds) => set({ activeSeeds: seeds }),

    // Function to immediately fetch latest mood state (for use during processing)
    fetchLatestMoodState: async () => {
      try {
        const apiService = (await import('../services/api')).apiService;
        const [moodResponse, emotionResponse] = await Promise.all([
          apiService.getCurrentMood(),
          apiService.getEmotionState(),
        ]);

        if (moodResponse.data) {
          get().updateCurrentMood(moodResponse.data);
        }

        if (emotionResponse.data) {
          get().updateEmotionState(emotionResponse.data);
          if (emotionResponse.data.mood_family) {
            get().setCurrentMoodFamily(emotionResponse.data.mood_family as MoodFamily);
          }
        }
      } catch (error) {
        console.error('❌ fetchLatestMoodState: Error updating mood state:', error);
      }
    },
  }))
);

// Subscribe to mood changes and update document data attribute
useDaemonStore.subscribe(
  (state) => state.currentMoodFamily,
  (moodFamily) => {
    document.documentElement.setAttribute('data-mood', moodFamily);
  }
);

export default useDaemonStore;
