// Chat messages display component with proper formatting

import type { FC, ReactNode } from 'react';
import { motion } from 'framer-motion';
import type { ConversationMessage } from '../../types/daemon';

interface ChatMessagesProps {
  messages: ConversationMessage[];
}

const ChatMessages: FC<ChatMessagesProps> = ({ messages }) => {
  // Render paragraphs as real elements so streaming newlines show immediately.
  // We split on double-newlines for paragraphs and preserve single newlines inside each paragraph.
  const renderContent = (content: string): ReactNode => {
    const paragraphs = content.split(/(?:\r?\n){2,}/g);
    return (
      <div>
        {paragraphs.map((para, idx) => (
          <p key={idx} className="whitespace-pre-wrap leading-relaxed mb-5 last:mb-0">
            {para}
          </p>
        ))}
      </div>
    );
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-[80%] rounded-lg px-4 py-3 ${
              message.role === 'user'
                ? 'bg-daemon-primary/30 border border-daemon-accent/50 text-slate-100'
                : 'bg-obsidian-100/50 border border-daemon-glow/30 text-slate-200'
            }`}
          >
            {/* Message header */}
            <div className="flex items-center justify-between mb-2">
              <span className={`text-xs font-medium ${
                message.role === 'user' ? 'text-daemon-accent' : 'text-daemon-glow'
              }`}>
                {message.role === 'user' ? 'YOU' : 'DAEMON'}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {formatTimestamp(message.timestamp)}
              </span>
            </div>

            {/* Message content */}
            <div className="prose prose-sm max-w-none prose-invert">
              {renderContent(message.content)}
            </div>

            {/* Token count if available */}
            {message.token_count > 0 && (
              <div className="mt-2 text-xs text-slate-500 font-mono">
                {message.token_count} tokens
              </div>
            )}

            {/* Emotional vector visualization for daemon messages */}
            {message.role === 'assistant' && message.self_affect && (
              <div className="mt-3 p-2 bg-obsidian-100/30 rounded border border-daemon-accent/20">
                <div className="text-xs text-daemon-glow mb-1">Emotional Resonance:</div>
                <div className="flex space-x-1">
                  {message.self_affect.slice(0, 8).map((value, i) => (
                    <div
                      key={i}
                      className="w-2 bg-daemon-glow/20 rounded-sm"
                      style={{
                        height: `${Math.max(2, Math.abs(value) * 20)}px`,
                        backgroundColor: value > 0 ? 'var(--daemon-glow)' : 'var(--daemon-accent)',
                        opacity: 0.3 + Math.abs(value) * 0.7,
                      }}
                      title={`Dimension ${i + 1}: ${value.toFixed(3)}`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default ChatMessages;
