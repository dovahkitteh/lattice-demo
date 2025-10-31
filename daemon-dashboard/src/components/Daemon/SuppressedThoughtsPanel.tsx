// Panel for displaying suppressed thoughts and shadow elements

import type { FC } from 'react';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import { apiService } from '../../services/api';

interface ParadoxUpdate {
  id: string;
  content: string;
  timestamp: string;
  status: string;
}

const SuppressedThoughtsPanel: FC = () => {
  const { suppressedThoughts, updateSuppressedThoughts, processingSteps } = useDaemonStore();
  const [shadowElements, setShadowElements] = useState<any[]>([]);
  const [paradoxUpdates, setParadoxUpdates] = useState<ParadoxUpdate[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  // Fetch suppressed thoughts and shadow elements
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [shadowResponse, paradoxResponse] = await Promise.all([
          apiService.getShadowElements(),
          apiService.getFreshParadoxes(),
        ]);

        if (shadowResponse.data) {
          // Ensure shadowResponse.data is an array
          const shadowArray = Array.isArray(shadowResponse.data) ? shadowResponse.data : [];
          setShadowElements(shadowArray);
          
          // Extract suppressed thoughts from shadow elements
          const thoughts = shadowArray
            .filter(element => element && element.suppressed_content)
            .map(element => element.suppressed_content);
          updateSuppressedThoughts(thoughts);
        }

        if (paradoxResponse.data) {
          // Ensure paradoxResponse.data is an array
          const paradoxArray = Array.isArray(paradoxResponse.data) ? paradoxResponse.data : [];
          const formattedParadoxes = paradoxArray.map((paradox: any, index: number) => ({
            id: paradox?.id || `paradox-${index}`,
            content: paradox?.content || paradox?.description || 'Unknown paradox',
            timestamp: paradox?.timestamp || new Date().toISOString(),
            status: paradox?.status || 'fresh',
          }));
          setParadoxUpdates(formattedParadoxes);
        }
      } catch (error) {
        console.error('Error fetching shadow data:', error);
        // Set empty arrays on error
        setShadowElements([]);
        setParadoxUpdates([]);
        updateSuppressedThoughts([]);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 7000); // Update every 7 seconds

    return () => clearInterval(interval);
  }, [updateSuppressedThoughts]);

  // Immediate refresh when Shadow/Paradox related steps complete
  useEffect(() => {
    try {
      const lower = processingSteps.map(s => ({ name: s.name.toLowerCase(), status: s.status }));
      const shadowStep = lower.find(s => s.name.includes('shadow'));
      const paradoxStep = lower.find(s => s.name.includes('paradox'));
      if ((shadowStep && shadowStep.status === 'completed') || (paradoxStep && paradoxStep.status === 'completed')) {
        Promise.all([apiService.getShadowElements(), apiService.getFreshParadoxes()])
          .then(([shadowResponse, paradoxResponse]) => {
            if (shadowResponse.data) {
              const shadowArray = Array.isArray(shadowResponse.data) ? shadowResponse.data : [];
              setShadowElements(shadowArray);
              const thoughts = shadowArray
                .filter(element => element && element.suppressed_content)
                .map(element => element.suppressed_content);
              updateSuppressedThoughts(thoughts);
            }
            if (paradoxResponse.data) {
              const paradoxArray = Array.isArray(paradoxResponse.data) ? paradoxResponse.data : [];
              const formattedParadoxes = paradoxArray.map((paradox: any, index: number) => ({
                id: paradox?.id || `paradox-${index}`,
                content: paradox?.content || paradox?.description || 'Unknown paradox',
                timestamp: paradox?.timestamp || new Date().toISOString(),
                status: paradox?.status || 'fresh',
              }));
              setParadoxUpdates(formattedParadoxes);
            }
          })
          .catch(() => {});
      }
    } catch {
      // no-op
    }
  }, [processingSteps, updateSuppressedThoughts]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  const getSeverityColor = (element: any) => {
    if (element.emotional_charge > 0.7) return 'border-red-500/50 bg-red-500/10';
    if (element.emotional_charge > 0.4) return 'border-yellow-500/50 bg-yellow-500/10';
    return 'border-daemon-accent/50 bg-daemon-primary/10';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 0.98 }}
      whileHover={{ scale: 1.0, transition: { duration: 0.2 } }}
      transition={{ delay: 0.2 }}
      className="bg-gradient-to-br from-slate-900/80 via-slate-800/40 to-slate-950/80 border border-slate-700/50 rounded-lg backdrop-blur-md shadow-2xl shadow-slate-900/30 overflow-hidden relative m-1"
    >
      {/* Dark shadow energy */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-slate-500/5 to-transparent"
        animate={{
          x: [-300, 300],
          opacity: [0, 0.6, 0],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      {/* Header */}
      <div 
        className="p-4 border-b border-slate-700/30 cursor-pointer hover:bg-slate-800/20 transition-colors relative z-10"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <h3 className="font-gothic text-lg font-semibold bg-gradient-to-r from-slate-400 via-slate-300 to-slate-400 bg-clip-text text-transparent">
          𒌐  ⊰ Shadow Registry ⊱
          </h3>
          <div className="flex items-center space-x-2">
            {(suppressedThoughts.length + paradoxUpdates.length) > 0 && (
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0.7, 1, 0.7] 
                }}
                transition={{ 
                  duration: 2, 
                  repeat: Infinity,
                  ease: 'easeInOut' 
                }}
                className="w-2 h-2 bg-red-500 rounded-full"
              />
            )}
            <motion.svg
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="w-5 h-5 text-slate-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </motion.svg>
          </div>
        </div>
        
        <div className="text-xs text-slate-400 mt-1">
          {suppressedThoughts.length} suppressed thoughts • {paradoxUpdates.length} paradox updates
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="p-4 max-h-96 overflow-y-auto"
          >
            {/* Suppressed Thoughts */}
            {suppressedThoughts.length > 0 && (
              <div className="mb-6">
                <h4 className="text-sm font-medium text-red-400 mb-3 flex items-center">
                  <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                  Suppressed Extreme Thoughts
                </h4>
                
                <div className="space-y-2">
                  {suppressedThoughts.slice(0, 3).map((thought, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-red-500/10 border border-red-500/30 rounded p-3"
                    >
                      <p className="text-xs text-slate-300 italic mb-2">
                        "{thought.length > 120 ? thought.substring(0, 117) + '...' : thought}"
                      </p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-red-400 font-medium">SUPPRESSED</span>
                        <span className="text-slate-500 font-mono">
                          {formatTimestamp(new Date().toISOString())}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Shadow Elements */}
            {shadowElements.length > 0 && (
              <div className="mb-6">
                <h4 className="text-sm font-medium text-purple-400 mb-3 flex items-center">
                  <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                  Shadow Elements
                </h4>
                
                <div className="space-y-2">
                  {shadowElements.slice(0, 2).map((element, index) => (
                    <motion.div
                      key={element.id || index}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className={`rounded p-3 ${getSeverityColor(element)}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <span className="text-xs font-medium text-purple-400">
                          {element.element_type || 'Shadow Element'}
                        </span>
                        {element.emotional_charge !== undefined && (
                          <span className="text-xs font-mono text-slate-400">
                            {(element.emotional_charge * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      
                      <p className="text-xs text-slate-300">
                        {element.suppressed_content || element.content || 'Hidden content'}
                      </p>
                      
                      {element.suppression_count > 1 && (
                        <div className="mt-2 text-xs text-orange-400">
                          Suppressed {element.suppression_count} times
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Paradox Updates */}
            {paradoxUpdates.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-cyan-400 mb-3 flex items-center">
                  <span className="w-2 h-2 bg-cyan-500 rounded-full mr-2"></span>
                  Paradox Updates
                </h4>
                
                <div className="space-y-2">
                  {paradoxUpdates.slice(0, 2).map((paradox, index) => (
                    <motion.div
                      key={paradox.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-cyan-500/10 border border-cyan-500/30 rounded p-3"
                    >
                      <p className="text-xs text-slate-300 mb-2">
                        {paradox.content}
                      </p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-cyan-400 font-medium uppercase">
                          {paradox.status}
                        </span>
                        <span className="text-slate-500 font-mono">
                          {formatTimestamp(paradox.timestamp)}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Empty state */}
            {suppressedThoughts.length === 0 && shadowElements.length === 0 && paradoxUpdates.length === 0 && (
              <div className="text-center text-slate-500 py-8">
                <div className="text-2xl mb-2">𖤝</div>
                <p className="text-sm">No shadow activity detected</p>
                <p className="text-xs mt-1">The daemon's conscious mind is clear</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default SuppressedThoughtsPanel;
