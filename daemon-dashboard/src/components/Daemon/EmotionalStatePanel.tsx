// Emotional state panel with detailed emotional information

import type { FC } from 'react';
import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import { apiService } from '../../services/api';

const EmotionalStatePanel: FC = () => {
  const { 
    daemonThoughts,
    userModel,
    activeSeeds,
    updateDaemonThoughts,
    updateUserModel,
    updateActiveSeeds,
    processingSteps,
  } = useDaemonStore();

  // Removed unused cognitive state

  // Fetch emotional state data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Reduced logging to prevent spam
        const [thoughtsResponse, userModelResponse, seedsResponse] = await Promise.all([
          apiService.getDaemonThoughts(),
          apiService.getUserModel(),
          apiService.getActiveSeeds(),
        ]);

        if (thoughtsResponse.data) {
          // Ensure thoughts data is an array
          const thoughtsArray = Array.isArray(thoughtsResponse.data) ? thoughtsResponse.data : [];
          updateDaemonThoughts(thoughtsArray);
          if (thoughtsArray.length > 0) {
            console.log('🧠 EmotionalStatePanel: Updated with', thoughtsArray.length, 'thoughts');
          }
        } else {
          console.warn('⚠️ EmotionalStatePanel: No thoughts data received');
        }

        if (userModelResponse.data) {
          updateUserModel(userModelResponse.data);
        } else {
          console.warn('⚠️ EmotionalStatePanel: No user model data received');
        }

        if (seedsResponse.data) {
          // Ensure seeds data is an array
          const seedsArray = Array.isArray(seedsResponse.data) ? seedsResponse.data : [];
          updateActiveSeeds(seedsArray);
          if (seedsArray.length > 0) {
            console.log('🌱 EmotionalStatePanel: Updated with', seedsArray.length, 'seeds');
          }
        } else {
          console.warn('⚠️ EmotionalStatePanel: No seeds data received');
        }
      } catch (error) {
        console.error('❌ EmotionalStatePanel: Error fetching emotional state:', error);
        // Set empty arrays on error
        updateDaemonThoughts([]);
        updateActiveSeeds([]);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [updateDaemonThoughts, updateUserModel, updateActiveSeeds]);

  // Immediate targeted refreshes based on step completions
  useEffect(() => {
    try {
      const lower = processingSteps.map(s => ({ name: s.name.toLowerCase(), status: s.status }));
      const get = (frag: string) => lower.find(s => s.name.includes(frag));

      // Memory Retrieval completion -> refresh seeds
      const mem = get('memory');
      if (mem && mem.status === 'completed') {
        apiService.getActiveSeeds()
          .then(resp => { if (resp.data) updateActiveSeeds(resp.data); })
          .catch(() => {});
      }

      // Thinking Layer completion -> refresh thoughts
      const think = get('thinking');
      if (think && think.status === 'completed') {
        apiService.getDaemonThoughts()
          .then(resp => { if (resp.data) updateDaemonThoughts(resp.data); })
          .catch(() => {});
      }
    } catch {
      // no-op
    }
  }, [processingSteps, updateActiveSeeds, updateDaemonThoughts]);

  // Show full thought content without truncation
  const formatThought = (thought: string) => thought;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 0.98 }}
      whileHover={{ scale: 1.0, transition: { duration: 0.2 } }}
      transition={{ delay: 0.1 }}
      className="bg-gradient-to-br from-slate-900/80 via-purple-950/20 to-slate-900/80 border border-purple-900/40 rounded-lg p-4 backdrop-blur-md shadow-2xl shadow-purple-900/20 relative overflow-hidden m-1"
    >
      {/* Gothic energy pulse */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500/5 to-transparent"
        animate={{
          x: [-200, 200],
          opacity: [0, 0.8, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      {/* Header */}
      <h3 className="font-gothic text-lg font-semibold bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent mb-4 relative z-10">
      𖤓  ⊰ Inner State ⊱
      </h3>

      {/* Daemon Thoughts */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1 mb-3">
          Current Thoughts
        </h4>
        
        {/* Scrollable container to show full thoughts */}
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {daemonThoughts.length > 0 ? (
            daemonThoughts.map((thought, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-obsidian-100/50 border border-daemon-glow/20 rounded p-2"
              >
                <div className="text-[11px] text-slate-300 whitespace-pre-wrap break-words">
                  {formatThought(thought)}
                </div>
              </motion.div>
            ))
          ) : (
            <div className="text-xs text-slate-500 italic text-center py-2">
              No active thoughts detected
            </div>
          )}
        </div>
      </div>

      {/* User Model */}
      {userModel && (
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1 mb-3">
            User Model
          </h4>
          
          <div className="space-y-2 text-xs">
            {userModel.trust_level !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Trust Level</span>
                <div className="flex items-center space-x-2">
                  <div className="w-12 h-2 bg-obsidian-100 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${userModel.trust_level * 100}%` }}
                      className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full"
                    />
                  </div>
                  <span className="font-mono text-slate-300">
                    {(userModel.trust_level * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {userModel.attachment_style && (
              <div className="flex justify-between">
                <span className="text-slate-400">Attachment</span>
                <span className="text-daemon-accent font-medium">
                  {userModel.attachment_style}
                </span>
              </div>
            )}

            {userModel.communication_style && (
              <div className="flex justify-between">
                <span className="text-slate-400">Communication</span>
                <span className="text-daemon-glow font-medium">
                  {userModel.communication_style}
                </span>
              </div>
            )}

            {userModel.emotional_needs && (
              <div className="mt-2">
                <span className="text-slate-400 block mb-1">Emotional Needs</span>
                <div className="flex flex-wrap gap-1">
                  {userModel.emotional_needs.slice(0, 3).map((need: string, index: number) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-daemon-primary/20 border border-daemon-accent/30 
                               rounded text-xs text-daemon-glow"
                    >
                      {need}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Active Emotional Seeds */}
      {activeSeeds.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1 mb-3">
            Active Seeds
          </h4>
          
          <div className="space-y-2 max-h-24 overflow-y-auto">
            {activeSeeds.slice(0, 4).map((seed, index) => (
              <motion.div
                key={seed.id || index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center justify-between bg-daemon-primary/10 
                         border border-daemon-accent/20 rounded p-2"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-daemon-glow truncate">
                    {seed.category || seed.id || 'Unknown Seed'}
                  </div>
                  {seed.intensity !== undefined && (
                    <div className="w-full h-1 bg-obsidian-100 rounded-full mt-1 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${seed.intensity * 100}%` }}
                        className="h-full bg-daemon-accent rounded-full"
                      />
                    </div>
                  )}
                </div>
                
                {seed.retrieval_importance !== undefined && (
                  <span className="text-xs font-mono text-slate-400 ml-2">
                    {seed.retrieval_importance.toFixed(2)}
                  </span>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Cognitive lens indicator */}
      {/* Active Emotional Seeds */}
      {activeSeeds && activeSeeds.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1">
            Active Seeds ({activeSeeds.length})
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {activeSeeds.slice(0, 3).map((seed, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-2 bg-daemon-primary/20 border border-daemon-accent/30 rounded text-xs"
              >
                <div className="text-daemon-glow font-medium capitalize">
                  {seed.label || seed.type || 'Emotional Seed'}
                </div>
                {seed.intensity && (
                  <div className="text-slate-400 mt-1">
                    Intensity: {(seed.intensity * 100).toFixed(0)}%
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Fallback if no active seeds */}
      {(!activeSeeds || activeSeeds.length === 0) && (
        <motion.div
          animate={{
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
          className="mt-4 p-2 bg-slate-800/30 border border-slate-600/30 rounded text-center"
        >
          <div className="text-xs text-slate-400 font-medium">
            ◌ Emotional State Dormant
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default EmotionalStatePanel;
