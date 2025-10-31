// Main chat area with void background and processing visualization

import type { FC } from 'react';
import { useRef, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import ChatMessages from './ChatMessages.tsx';
import ChatInput from './ChatInput.tsx';
import ProcessingSequence from './ProcessingSequence.tsx';

const ChatArea: FC = () => {
  const { activeConversation, isProcessing, processingSteps } = useDaemonStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevMessageCountRef = useRef<number>(0);
  
  // Stable ambient particle positions and timings
  const particles = useMemo(
    () =>
      Array.from({ length: 12 }).map((_, idx) => ({
        key: `p-${idx}-${Math.random().toString(36).slice(2, 7)}`,
        left: Math.random() * 100,
        top: Math.random() * 100,
        duration: 4 + Math.random() * 4,
        delay: idx * 0.5,
      })),
    []
  );

  // Auto-scroll only when a new message is appended (message count increases).
  // Streaming updates that modify the last message's content should not force-scroll.
  useEffect(() => {
    const messages = activeConversation?.messages || [];
    const prevCount = prevMessageCountRef.current;
    const currentCount = messages.length;

    const messageAppended = currentCount > prevCount;
    if (autoScroll && messageAppended) {
      // Use smooth scroll when appending, but avoid repeatedly forcing during streaming
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }

    prevMessageCountRef.current = currentCount;
  }, [activeConversation?.messages, autoScroll]);

  // Toggle autoScroll based on user scroll position
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const onScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
      setAutoScroll(nearBottom);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll as any);
  }, []);

  return (
    <div className="h-full max-h-[85vh] min-h-0 flex flex-col bg-obsidian-100/20 border border-daemon-accent/30 rounded-lg overflow-hidden">
      {/* Chat header with context info */}
      <div className="border-b border-daemon-accent/20 p-4 bg-obsidian-100/40">
        <div className="flex items-center justify-between">
          <h3 className="font-gothic text-lg font-semibold text-daemon-glow">
            {activeConversation ? activeConversation.title : 'Select a conversation'}
          </h3>
          
          {activeConversation && (
            <div className="text-sm text-slate-400 space-x-4">
              <span>
                Context: {activeConversation.total_tokens}/8192 tokens 
                ({Math.round((activeConversation.total_tokens / 8192) * 100)}%)
              </span>
              <span>
                {Array.isArray(activeConversation.messages) ? activeConversation.messages.length : 0} messages
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Main chat area with void background */}
      <div className="flex-1 min-h-0 relative overflow-hidden">
        {/* Ambient background: single gradient that breathes. Colors change via CSS variables, no positional jumps. */}
        <div className="absolute inset-0 pointer-events-none">
          <motion.div
            className="absolute inset-0"
            style={{
              background:
                'radial-gradient(circle at 50% 50%, var(--daemon-primary) 0%, transparent 70%)',
            }}
            animate={{ opacity: [0.35, 0.55, 0.35] }}
            transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          />
          
          {/* Floating particles with stable positions to avoid jumps */}
          {particles.map((p) => (
            <motion.div
              key={p.key}
              className="absolute w-1 h-1 bg-daemon-glow/40 rounded-full"
              style={{ left: `${p.left}%`, top: `${p.top}%` }}
              animate={{ y: [-20, 20, -20], x: [-10, 10, -10], opacity: [0.25, 0.8, 0.25] }}
              transition={{ duration: p.duration, repeat: Infinity, ease: 'easeInOut', delay: p.delay }}
            />
          ))}
        </div>

        {/* Messages container */}
        <div className="relative z-10 h-full min-h-0 flex flex-col">
          <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
            {activeConversation && Array.isArray(activeConversation.messages) ? (
              <>
                <ChatMessages messages={activeConversation.messages} />
                {(isProcessing || (processingSteps && processingSteps.length > 0)) && <ProcessingSequence />}
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center text-slate-400"
                >
                  <div className="text-6xl mb-4 font-gothic">⸸</div>
                  <p className="text-lg font-gothic">
                    Begin conversation to witness recursive evolution
                  </p>
                  <p className="text-sm mt-2">
                    Create a new conversation or select an existing one to start
                  </p>
                </motion.div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Chat input area */}
      {activeConversation && (
        <div className="border-t border-daemon-accent/20 bg-obsidian-100/40">
          <ChatInput />
        </div>
      )}
    </div>
  );
};

export default ChatArea;
