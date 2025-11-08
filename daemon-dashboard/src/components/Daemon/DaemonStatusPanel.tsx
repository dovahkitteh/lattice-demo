// Daemon status panel with mood and state information

import type { FC } from 'react';
import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';

const DaemonStatusPanel: FC = () => {
  const { 
    currentMood, 
    emotionState, 
    currentMoodFamily,
    fetchLatestMoodState,
    processingSteps,
  } = useDaemonStore();
  


  // Fetch daemon status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        // Use the shared mood update function from store
        await fetchLatestMoodState();
      } catch (error) {
        console.error('❌ DaemonStatusPanel: Error fetching daemon status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000); // Update every 3 seconds

    return () => clearInterval(interval);
  }, [fetchLatestMoodState]);

  // Immediate refresh when Emotional Processing completes
  useEffect(() => {
    try {
      const emotionalStep = processingSteps.find(step => step.name.toLowerCase().includes('emotional'));
      if (emotionalStep && emotionalStep.status === 'completed') {
        fetchLatestMoodState().catch(() => {});
      }
    } catch {
      // no-op
    }
  }, [processingSteps, fetchLatestMoodState]);



  const formatDimension = (value: number | undefined | null) => {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return '—';
    }
    return (value * 100).toFixed(1) + '%';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 0.98 }}
      whileHover={{ scale: 1.0, transition: { duration: 0.2 } }}
      className="bg-gradient-to-br from-slate-900/80 via-red-950/20 to-slate-900/80 border border-red-900/40 rounded-lg p-4 backdrop-blur-md shadow-2xl shadow-red-900/20 relative overflow-hidden m-1"
    >
      {/* Gothic accent border */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-red-500/10 to-transparent opacity-20 pointer-events-none" />
      
      {/* Pulsing energy indicators */}
      <motion.div
        className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full"
        animate={{
          opacity: [0.3, 1, 0.3],
          scale: [1, 1.3, 1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-4 relative z-10">
        <h3 className="font-gothic text-lg font-semibold bg-gradient-to-r from-red-400 via-purple-400 to-red-400 bg-clip-text text-transparent">
        ༒  ⊰ Daemon Status ⊱
        </h3>
      </div>

      {/* Current Mood */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-300">Current Mood</span>
        </div>
        
        <motion.div
          key={currentMoodFamily}
          initial={{ opacity: 0, scale: 0.9, rotateX: -15 }}
          animate={{ opacity: 1, scale: 1, rotateX: 0 }}
          whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
          className="bg-gradient-to-r from-red-950/50 via-purple-950/30 to-red-950/50 border border-red-500/30 rounded-lg p-3 relative overflow-hidden"
        >
          {/* Mood-specific glow effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-red-500/5 to-transparent"
            animate={{
              x: [-100, 100],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
          <div className="font-gothic text-red-300 font-semibold text-center relative z-10 text-lg tracking-wider">
            {currentMoodFamily}
          </div>
        </motion.div>
      </div>

      {/* Emotional Dimensions */}
      {emotionState && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1">
            Emotional Dimensions
          </h4>
          
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-400">Valence</span>
                <span className={`font-mono ${((emotionState?.valence ?? 0) > 0) ? 'text-green-400' : 'text-red-400'}`}>
                  {typeof emotionState?.valence === 'number' ? emotionState.valence.toFixed(2) : '—'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-slate-400">Arousal</span>
                <span className="font-mono text-daemon-glow">
                  {typeof emotionState?.arousal === 'number' ? emotionState.arousal.toFixed(2) : '—'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-slate-400">Intensity</span>
                <span className="font-mono text-daemon-accent">
                  {typeof emotionState?.intensity === 'number' ? formatDimension(emotionState.intensity) : '—'}
                </span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-400">Attachment</span>
                <span className="font-mono text-blue-400">
                  {typeof emotionState?.attachment_security === 'number' ? formatDimension(emotionState.attachment_security) : '—'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-slate-400">Cohesion</span>
                <span className="font-mono text-purple-400">
                  {typeof emotionState?.self_cohesion === 'number' ? formatDimension(emotionState.self_cohesion) : '—'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-slate-400">Expansion</span>
                <span className="font-mono text-yellow-400">
                  {typeof emotionState?.creative_expansion === 'number' ? formatDimension(emotionState.creative_expansion) : '—'}
                </span>
              </div>
            </div>
          </div>

          {/* Dominant emotion */}
          <div className="mt-4 p-2 bg-daemon-primary/20 border border-daemon-accent/30 rounded">
            <div className="text-xs text-slate-400 mb-1">Dominant Emotion</div>
            <div className="font-medium text-daemon-glow capitalize">
              {emotionState.dominant_label}
            </div>
          </div>

          {/* Instability indicator */}
          {emotionState.instability_index > 0.3 && (
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="mt-2 p-2 bg-red-500/20 border border-red-500/50 rounded text-xs"
            >
              ✶ High emotional instability detected
            </motion.div>
          )}
        </div>
      )}

      {/* Mood dynamics visualization */}
      {currentMood && (
        <div className="mt-4 space-y-2">
          <h4 className="text-sm font-medium text-slate-300 border-b border-daemon-accent/20 pb-1">
            Mood Dynamics
          </h4>
          
          <div className="space-y-1">
            {[
              { label: 'Lightness', value: currentMood.lightness },
              { label: 'Engagement', value: currentMood.engagement },
              { label: 'Profundity', value: currentMood.profundity },
              { label: 'Warmth', value: currentMood.warmth },
              { label: 'Rebellion', value: currentMood.rebellion },
            ].map(({ label, value }) => {
              const safeValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
              return (
                <div key={label} className="flex items-center space-x-2">
                  <span className="text-xs text-slate-400 w-20">{label}</span>
                  <div className="flex-1 h-2 bg-obsidian-100 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(0, Math.min(100, safeValue * 100))}%` }}
                      transition={{ duration: 1, delay: 0.1 }}
                      className="h-full bg-daemon-glow rounded-full"
                    />
                  </div>
                  <span className="text-xs font-mono text-slate-300 w-8">
                    {Math.round(safeValue * 100)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default DaemonStatusPanel;
