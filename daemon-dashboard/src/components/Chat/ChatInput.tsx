// Chat input component with streaming support

import type { FC } from 'react';
import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import { apiService } from '../../services/api';

const ChatInput: FC = () => {
  const {
    currentMessage,
    setCurrentMessage,
    activeConversation,
    addMessage,
    isProcessing,
    setIsProcessing,
    setProcessingSteps,
    updateActiveSeeds,
    updateDaemonThoughts,
    updateUserModel,
    fetchLatestMoodState,
  } = useDaemonStore();

  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const emotionalProcessingCompleted = useRef<boolean>(false);
  const lastProcessingStepsRef = useRef<string>('');
  const prevStepStatusRef = useRef<Record<string, string>>({});
  const isStreamingRef = useRef<boolean>(false);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [currentMessage]);

  // Clean up polling when conversation changes
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [activeConversation?.session_id]);

  const sendMessage = async () => {
    if (!currentMessage.trim() || !activeConversation || isProcessing) return;

    const userMessage = currentMessage.trim();
    setCurrentMessage('');
    setIsProcessing(true);
    isStreamingRef.current = false;

    // Add user message immediately
    addMessage({ role: 'user', content: userMessage });

    // Initialize empty processing steps - they'll be populated by the backend
    setProcessingSteps([]);
    // Reset emotional processing completion flag and step tracking
    emotionalProcessingCompleted.current = false;
    lastProcessingStepsRef.current = '';

    try {
      // Start polling for processing status immediately
      let pollCount = 0;
      const maxPolls = 300; // Max ~60 seconds of polling
      
      // Clear any existing polling interval
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      
      pollIntervalRef.current = setInterval(async () => {
        try {
          pollCount++;
          const statusResponse = await apiService.getProcessingStatus();
          if (statusResponse.data) {
            const { processing_steps } = statusResponse.data;
            // Preserve last non-empty steps; backend may auto-clear after a few seconds
            if (Array.isArray(processing_steps) && processing_steps.length > 0) {
              // Only update if steps actually changed to prevent flickering
              const stepsSignature = JSON.stringify(processing_steps.map(s => ({ name: s.name, status: s.status })));
              if (stepsSignature !== lastProcessingStepsRef.current) {
                lastProcessingStepsRef.current = stepsSignature;
                // Keep processing steps updated for listeners even during streaming
                setProcessingSteps(processing_steps);
              
                // Check if "Emotional Processing" step just completed
                const emotionalStep = processing_steps.find(step => 
                  step.name === 'Emotional Processing' || step.name.toLowerCase().includes('emotional')
                );
                
                if (emotionalStep && emotionalStep.status === 'completed' && !emotionalProcessingCompleted.current) {
                  emotionalProcessingCompleted.current = true;
                  console.log('🎭 ChatInput: Emotional Processing completed, updating mood state immediately...');
                  // Trigger immediate mood update
                  fetchLatestMoodState().catch(error => 
                    console.error('Error updating mood after emotional processing:', error)
                  );
                }

                // Detect per-step completion transitions and trigger targeted refreshes
                try {
                  const currentStatusMap: Record<string, string> = {};
                  for (const step of processing_steps) {
                    currentStatusMap[step.name.toLowerCase()] = step.status;
                  }

                  const justCompleted = (nameFragment: string) => {
                    const now = Object.entries(currentStatusMap).find(([k]) => k.includes(nameFragment))?.[1];
                    const prev = Object.entries(prevStepStatusRef.current).find(([k]) => k.includes(nameFragment))?.[1];
                    return now === 'completed' && prev !== 'completed';
                  };

                  // Emotional Processing -> refresh mood immediately (already handled above, keep as safety)
                  if (justCompleted('emotional')) {
                    fetchLatestMoodState().catch(() => {});
                  }

                  // Memory Retrieval -> refresh active seeds immediately
                  if (justCompleted('memory')) {
                    apiService.getActiveSeeds()
                      .then(resp => { if (resp.data) updateActiveSeeds(resp.data); })
                      .catch(err => console.error('Error refreshing active seeds after Memory Retrieval:', err));
                  }

                  // Thinking Layer -> refresh daemon thoughts and user model immediately
                  if (justCompleted('thinking')) {
                    apiService.getDaemonThoughts()
                      .then(resp => { if (resp.data) updateDaemonThoughts(resp.data); })
                      .catch(err => console.error('Error refreshing thoughts after Thinking Layer:', err));
                    apiService.getUserModel()
                      .then(resp => { if (resp.data) updateUserModel(resp.data); })
                      .catch(err => console.error('Error refreshing user model after Thinking Layer:', err));
                  }

                  // Stop polling when response generation completes
                  const responseStep = processing_steps.find(step => step.name.toLowerCase().includes('response'));
                  const generationStep = processing_steps.find(step => step.name.toLowerCase().includes('generation'));
                  if ((responseStep && responseStep.status === 'completed') || (generationStep && generationStep.status === 'completed')) {
                    if (pollIntervalRef.current) {
                      clearInterval(pollIntervalRef.current);
                      pollIntervalRef.current = null;
                    }
                  }

                  // Update prev map for next comparison
                  prevStepStatusRef.current = currentStatusMap;
                } catch (cmpErr) {
                  console.warn('Processing step comparison error:', cmpErr);
                }
              }
            }
            
            // Keep polling to capture late step completions
            if (pollCount >= maxPolls) {
              if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
              }
            }
          }
        } catch (error) {
          console.error('Error polling processing status:', error);
        }
      }, 200); // Poll every 200ms for better responsiveness

      // Send to API - this will trigger the real backend processing
      const messages = [
        ...activeConversation.messages,
        { role: 'user', content: userMessage },
      ];

      const response = await apiService.sendMessage(messages);
      
      // Check if response is streaming or JSON
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        // Non-streaming response - parse JSON directly
        const jsonResponse = await response.json();
        const assistantMessage = jsonResponse.choices?.[0]?.message?.content || '';
        
        if (assistantMessage) {
          addMessage({ role: 'assistant', content: assistantMessage });
        }

        // End processing now that a final response arrived
        setIsProcessing(false);
        setProcessingSteps([]);
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } else {
        // Streaming response with real-time updates
        let assistantMessage = '';
        let streamingMessageId: string | null = null;
        
        for await (const chunk of apiService.parseStreamResponse(response)) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            if (!isStreamingRef.current) {
              isStreamingRef.current = true;
              // Hide processing sequence as soon as the first token arrives
              setIsProcessing(false);
              setProcessingSteps([]);
              // DO NOT clear the polling interval; keep it running to catch later step completions
            }
            assistantMessage += content;
            
            // Real-time streaming: update the message as it arrives
            if (!streamingMessageId) {
              // Add initial message with first content
              addMessage({ role: 'assistant', content: assistantMessage });
              streamingMessageId = 'streaming';
            } else {
              // Update the existing streaming message
              const { activeConversation } = useDaemonStore.getState();
              if (activeConversation && activeConversation.messages.length > 0) {
                const lastMessage = activeConversation.messages[activeConversation.messages.length - 1];
                if (lastMessage.role === 'assistant') {
                  // Update the last assistant message with accumulated content
                  const updatedConversation = {
                    ...activeConversation,
                    messages: [
                      ...activeConversation.messages.slice(0, -1),
                      { ...lastMessage, content: assistantMessage }
                    ],
                    last_updated: new Date().toISOString()
                  };
                  useDaemonStore.getState().setActiveConversation(updatedConversation);
                }
              }
            }
          }
          
          // Check for finish_reason to ensure we capture the complete response
          if (chunk.choices[0]?.finish_reason) {
            break;
          }
        }
        
        // Ensure final message is properly stored (in case of any streaming issues)
        if (assistantMessage && streamingMessageId) {
          const { activeConversation } = useDaemonStore.getState();
          if (activeConversation && activeConversation.messages.length > 0) {
            const lastMessage = activeConversation.messages[activeConversation.messages.length - 1];
            if (lastMessage.role === 'assistant' && lastMessage.content !== assistantMessage) {
              // Final update to ensure complete message is stored
              const updatedConversation = {
                ...activeConversation,
                messages: [
                  ...activeConversation.messages.slice(0, -1),
                  { ...lastMessage, content: assistantMessage }
                ],
                last_updated: new Date().toISOString()
              };
              useDaemonStore.getState().setActiveConversation(updatedConversation);
            }
          }
        } else if (assistantMessage && !streamingMessageId) {
          // Fallback: add complete message if streaming setup failed
          addMessage({ role: 'assistant', content: assistantMessage });
        }
      }

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      addMessage({ 
        role: 'assistant', 
        content: 'I apologize, but I encountered an error processing your message. Please try again.' 
      });
    } finally {
      // Note: intervals are cleaned up when streaming begins or final JSON arrives
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="p-4 space-y-3">
      {/* Message input */}
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={currentMessage}
          onChange={(e) => setCurrentMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          placeholder="Speak with the daemon..."
          disabled={isProcessing}
          rows={1}
          className="w-full resize-none bg-obsidian-100/50 border border-daemon-accent/50 
                     rounded-lg px-4 py-3 pr-12 text-slate-100 placeholder-slate-400
                     focus:outline-none focus:border-daemon-glow focus:ring-1 focus:ring-daemon-glow
                     disabled:opacity-50 disabled:cursor-not-allowed
                     min-h-[48px] max-h-32 overflow-y-auto"
        />

        {/* Send button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={sendMessage}
          disabled={!currentMessage.trim() || isProcessing}
          className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 
                     bg-daemon-primary/50 hover:bg-daemon-primary/70 
                     border border-daemon-accent/50 hover:border-daemon-glow
                     rounded-lg flex items-center justify-center
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-all duration-200"
        >
          {isProcessing ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              className="w-4 h-4 border-2 border-daemon-accent border-t-transparent rounded-full"
            />
          ) : (
            <svg className="w-4 h-4 text-daemon-glow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )}
        </motion.button>
      </div>

      {/* Input hints */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>
          Press Enter to send, Shift+Enter for new line
        </span>
        <span className="font-mono">
          {currentMessage.length} characters
        </span>
      </div>
    </div>
  );
};

export default ChatInput;
