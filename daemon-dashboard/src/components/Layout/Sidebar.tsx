// Sidebar with conversation history and management

import type { FC } from 'react';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import { apiService } from '../../services/api';
import type { ConversationSession } from '../../types/daemon';

const Sidebar: FC = () => {
  const {
    conversations,
    activeConversation,
    updateConversations,
    setActiveConversation,
  } = useDaemonStore();

  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    try {
      const response = await apiService.getConversationSessions();
      if (response.data) {
        // API returns the array directly per backend `get_conversation_sessions`
        const conversationsArray: ConversationSession[] = Array.isArray(response.data)
          ? response.data
          : Array.isArray((response.data as any).sessions)
            ? (response.data as any).sessions
            : [];

        updateConversations(conversationsArray);

        // Non-blocking: enrich with message counts by fetching session details
        // Do this in the background to avoid delaying initial render
        Promise.allSettled(
          conversationsArray.map((s) => apiService.getConversationSession(s.session_id))
        ).then((results) => {
          const byId: Record<string, ConversationSession> = {};
          results.forEach((res) => {
            if (res.status === 'fulfilled' && res.value.data) {
              const detailed = res.value.data as ConversationSession;
              byId[detailed.session_id] = detailed;
            }
          });
          if (Object.keys(byId).length > 0) {
            const merged = conversationsArray.map((s) =>
              byId[s.session_id]
                ? { ...s, messages: byId[s.session_id].messages }
                : s
            );
            updateConversations(merged);
          }
        });

        if (!activeConversation && conversationsArray.length > 0) {
          const active = conversationsArray.find((c: ConversationSession) => c.is_active) || conversationsArray[0];
          setActiveConversation(active);
        }
      }
    } catch (error) {
      console.error('Error loading conversations:', error);
      // Set empty array if API fails
      updateConversations([]);
    }
  };

  const createNewSession = async () => {
    setIsCreating(true);
    try {
      const response = await apiService.createNewSession();
      if (response.data) {
        await loadConversations();
        setActiveConversation(response.data);
      }
    } finally {
      setIsCreating(false);
    }
  };

  const selectSession = async (session: ConversationSession) => {
    // Set as active session
    await apiService.setActiveSession(session.session_id);
    setActiveConversation(session);
  };

  const startEdit = (session: ConversationSession) => {
    setEditingSession(session.session_id);
    setEditTitle(session.title);
  };

  const saveEdit = async () => {
    if (editingSession && editTitle.trim()) {
      await apiService.renameSession(editingSession, editTitle.trim());
      await loadConversations();
    }
    setEditingSession(null);
    setEditTitle('');
  };

  const cancelEdit = () => {
    setEditingSession(null);
    setEditTitle('');
  };

  const deleteSession = async (sessionId: string) => {
    if (confirm('Are you sure you want to delete this conversation?')) {
      await apiService.deleteSession(sessionId);
      await loadConversations();
      
      // If deleted session was active, select another
      if (activeConversation?.session_id === sessionId) {
        const remaining = conversations.filter(c => c.session_id !== sessionId);
        setActiveConversation(remaining.length > 0 ? remaining[0] : null);
      }
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className="h-full min-h-0 flex flex-col bg-obsidian-100/30 border-r border-daemon-accent/20">
      {/* Header */}
      <div className="p-4 border-b border-daemon-accent/20">
        <h2 className="text-xl font-gothic font-semibold text-daemon-glow mb-4">
          Conversations
        </h2>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={createNewSession}
          disabled={isCreating}
          className="w-full py-2 px-4 bg-daemon-primary/30 hover:bg-daemon-primary/50 
                     border border-daemon-accent/50 rounded-lg transition-colors
                     text-sm font-medium disabled:opacity-50"
        >
          {isCreating ? (
            <span className="flex items-center justify-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="w-4 h-4 border-2 border-daemon-accent border-t-transparent rounded-full mr-2"
              />
              Creating...
            </span>
          ) : (
            '+ New Conversation'
          )}
        </motion.button>
      </div>

      {/* Conversations list */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <AnimatePresence>
          {conversations.map((session, index) => (
            <motion.div
              key={session.session_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ delay: index * 0.05 }}
              className={`group relative border-b border-daemon-accent/10 ${
                activeConversation?.session_id === session.session_id
                  ? 'bg-daemon-primary/20 border-l-4 border-l-daemon-accent'
                  : 'hover:bg-daemon-primary/10'
              }`}
            >
              <div className="p-3">
                {editingSession === session.session_id ? (
                  <div className="space-y-2">
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      className="w-full px-2 py-1 bg-obsidian-100 border border-daemon-accent/50 
                               rounded text-sm focus:outline-none focus:border-daemon-glow"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') saveEdit();
                        if (e.key === 'Escape') cancelEdit();
                      }}
                    />
                    <div className="flex space-x-2">
                      <button
                        onClick={saveEdit}
                        className="px-2 py-1 bg-daemon-accent/30 hover:bg-daemon-accent/50 
                                 rounded text-xs transition-colors"
                      >
                        Save
                      </button>
                      <button
                        onClick={cancelEdit}
                        className="px-2 py-1 bg-red-500/30 hover:bg-red-500/50 
                                 rounded text-xs transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div
                      onClick={() => selectSession(session)}
                      className="cursor-pointer"
                    >
                      <h3 className="font-medium text-sm line-clamp-2 mb-1">
                        {session.title}
                      </h3>
                      <div className="text-xs text-slate-400 flex items-center justify-between">
                        <span>{formatDate(session.last_updated)}</span>
                        <span>{Array.isArray(session.messages) ? session.messages.length : (session as any).message_count ?? 0} msgs</span>
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 
                                  transition-opacity flex space-x-1">
                      <button
                        onClick={() => startEdit(session)}
                        className="p-1 hover:bg-daemon-accent/30 rounded text-xs"
                        title="Rename"
                      >
                        ✏️
                      </button>
                      <button
                        onClick={() => deleteSession(session.session_id)}
                        className="p-1 hover:bg-red-500/30 rounded text-xs"
                        title="Delete"
                      >
                        🗑️
                      </button>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {conversations.length === 0 && (
          <div className="p-4 text-center text-slate-400">
            <p className="text-sm">No conversations yet.</p>
            <p className="text-xs mt-1">Start a new conversation to begin.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
