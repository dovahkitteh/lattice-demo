// Beautiful processing sequence visualization

import type { FC } from 'react';
import { useMemo, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';

const ProcessingSequence: FC = () => {
  const { processingSteps } = useDaemonStore();
  const [isVisible, setIsVisible] = useState(true);

  // Add a small delay before hiding to prevent quick flashes
  useEffect(() => {
    if (processingSteps.length === 0) {
      const hideTimeout = setTimeout(() => setIsVisible(false), 500);
      return () => clearTimeout(hideTimeout);
    } else {
      setIsVisible(true);
    }
  }, [processingSteps.length]);

  // Note: do not return early before hooks; compute visibility and check later

  const defaultSteps = [
    { name: 'Input Analysis', description: 'Parsing user input and context...', status: 'processing' as const },
    { name: 'Emotional Processing', description: 'Analyzing emotional triggers and state...', status: 'pending' as const },
    { name: 'Thinking Layer', description: 'Deep reasoning and reflection...', status: 'pending' as const },
    { name: 'Response Generation', description: 'Crafting response with emotional context...', status: 'pending' as const },
  ];

  // Memoize steps to prevent flickering from identical updates
  const steps = useMemo(() => {
    const rawSteps = processingSteps.length > 0 ? processingSteps : defaultSteps;
    return rawSteps.filter(step => step.name !== 'Memory Retrieval');
  }, [processingSteps]);

  // Create stable keys by combining name and status to prevent flicker
  const stableSteps = useMemo(() => {
    return steps.map((step, index) => ({
      ...step,
      stableKey: `${step.name}-${index}`, // Use index for stability rather than status
    }));
  }, [steps]);

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'processing':
        return '⚡';
      case 'error':
        return '✗';
      default:
        return '○';
    }
  };

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'processing':
        return 'text-daemon-glow';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-slate-500';
    }
  };

  // After all hooks are declared, decide whether to render
  const shouldRender = isVisible || processingSteps.length > 0;
  if (!shouldRender) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-obsidian-100/60 border border-daemon-accent/30 rounded-lg p-4 backdrop-blur-sm"
    >
      <div className="flex items-center mb-3">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="w-5 h-5 border-2 border-daemon-accent border-t-transparent rounded-full mr-3"
        />
        <h4 className="font-gothic text-daemon-glow font-semibold">
          DAEMON PROCESSING
        </h4>
      </div>

      <div className="space-y-3">
        <AnimatePresence mode="popLayout">
          {stableSteps.map((step, index) => (
            <motion.div
              key={step.stableKey}
              layout
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ 
                delay: index * 0.1,
                layout: { duration: 0.3 },
                opacity: { duration: 0.2 }
              }}
              className="flex items-center space-x-3"
            >
              <motion.div
                animate={step.status === 'processing' ? {
                  scale: [1, 1.2, 1],
                  boxShadow: [
                    '0 0 0 0 var(--daemon-glow)',
                    '0 0 0 8px transparent'
                  ]
                } : {}}
                transition={{
                  duration: 1.5,
                  repeat: step.status === 'processing' ? Infinity : 0,
                  ease: 'easeOut'
                }}
                className={`w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs font-bold
                  ${step.status === 'processing' 
                    ? 'border-daemon-glow bg-daemon-glow/20' 
                    : step.status === 'completed'
                    ? 'border-green-400 bg-green-400/20'
                    : step.status === 'error'
                    ? 'border-red-400 bg-red-400/20'
                    : 'border-slate-500 bg-slate-500/10'
                  }`}
              >
                <span className={getStepColor(step.status)}>
                  {getStepIcon(step.status)}
                </span>
              </motion.div>

              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className={`font-medium text-sm ${
                    step.status === 'processing' ? 'text-daemon-glow' : 'text-slate-300'
                  }`}>
                    {step.name}
                  </span>
                  {'duration' in step && step.duration && (
                    <span className="text-xs text-slate-400 font-mono">
                      {step.duration}ms
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {step.description}
                </p>
              </div>

              {/* Animated connection line */}
              {index < stableSteps.length - 1 && (
                <motion.div
                  className="absolute left-[11px] w-0.5 h-6 bg-daemon-accent/30"
                  style={{ 
                    top: '36px',
                    transformOrigin: 'top'
                  }}
                  animate={{
                    scaleY: step.status === 'completed' ? 1 : 0.3,
                    opacity: step.status === 'completed' ? 1 : 0.3,
                  }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Processing glow effect */}
      <motion.div
        className="absolute inset-0 rounded-lg pointer-events-none"
        animate={{
          boxShadow: [
            '0 0 20px var(--daemon-glow)',
            '0 0 40px var(--daemon-glow)',
            '0 0 20px var(--daemon-glow)',
          ],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </motion.div>
  );
};

export default ProcessingSequence;
