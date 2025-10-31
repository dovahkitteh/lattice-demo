import type { FC } from 'react';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { apiService } from '../../services/api';
import { useDaemonStore } from '../../stores/daemon-store';

interface UserModelData {
  unified_user_model?: {
    core_model?: {
      narrative_belief?: string;
      trust_level?: number;
      perceived_distance?: number;
      attachment_anxiety?: number;
      cognitive_processing_style?: string;
      communication_preferences?: string[];
      emotional_vulnerability_patterns?: string[];
      interaction_energy_level?: number;
      growth_edge_areas?: string[];
    };
    components?: Array<{
      component_id: string;
      category: string;
      title: string;
      description: string;
      confidence: number;
      emotional_significance: number;
      stability: number;
    }>;
    recent_analyses?: Array<{
      insights: Array<{
        category: string;
        description: string;
        confidence: number;
        emotional_charge: number;
        potential_misunderstanding: boolean;
        subject?: string;
      }>;
    }>;
  };
}

const UserInsightsPanel: FC = () => {
  const [data, setData] = useState<UserModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['core']));
  const { processingSteps } = useDaemonStore();

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const resp = await apiService.getUserModelDetailed();
        if (resp.error) {
          setError(resp.error);
        } else {
          setData(resp.data);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 8000); // Update every 8 seconds
    return () => clearInterval(interval);
  }, []);

  // Immediate refresh when Thinking Layer completes
  useEffect(() => {
    try {
      const thinking = processingSteps.find(s => s.name.toLowerCase().includes('thinking'));
      if (thinking && thinking.status === 'completed') {
        apiService.getUserModelDetailed()
          .then(resp => { if (!resp.error) setData(resp.data); })
          .catch(() => {});
      }
    } catch {
      // no-op
    }
  }, [processingSteps]);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const coreModel = data?.unified_user_model?.core_model;
  const components = data?.unified_user_model?.components?.slice(0, 3) || []; // Show only top 3 most recent
  const recentInsights = data?.unified_user_model?.recent_analyses?.[0]?.insights?.slice(0, 2) || [];

  const formatMetric = (value: number | undefined, suffix: string = '') => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return `${(value * 100).toFixed(0)}${suffix}`;
  };

  const getMetricColor = (value: number | undefined, invert: boolean = false) => {
    if (typeof value !== 'number') return 'text-slate-400';
    const threshold = invert ? 0.3 : 0.7;
    const isGood = invert ? value < threshold : value > threshold;
    return isGood ? 'text-emerald-400' : value > 0.5 ? 'text-yellow-400' : 'text-red-400';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 0.98 }}
      whileHover={{ scale: 1.0, transition: { duration: 0.2 } }}
      className="bg-obsidian-100/30 border border-daemon-accent/30 rounded-lg backdrop-blur-sm overflow-hidden m-1"
    >
      {/* Header */}
      <div className="p-4 border-b border-daemon-accent/20">
        <div className="flex items-center justify-between">
          <h3 className="font-gothic text-lg font-semibold text-daemon-glow flex items-center gap-2">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              className="w-5 h-5 border-2 border-daemon-accent border-t-transparent rounded-full"
            />
            𖤍  ⊰ User Psyche ⊱
          </h3>
          {coreModel && (
            <div className="text-xs text-slate-400 font-mono space-x-2">
              <span className={`${getMetricColor(coreModel.trust_level)}`}>
                Trust {formatMetric(coreModel.trust_level, '%')}
              </span>
              <span className={`${getMetricColor(coreModel.attachment_anxiety, true)}`}>
                Anx {formatMetric(coreModel.attachment_anxiety, '%')}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="p-4 space-y-4 max-h-96 overflow-y-auto scrollbar-thin scrollbar-track-obsidian-100/20 scrollbar-thumb-daemon-accent/40">


        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-sm text-red-400 p-3 bg-red-500/10 border border-red-500/30 rounded"
          >
            {error}
          </motion.div>
        )}

        {coreModel && (
          <>
            {/* Core Psychological Profile */}
            <motion.div
              layout
              className="bg-obsidian-100/40 border border-daemon-accent/20 rounded-lg overflow-hidden"
            >
              <div
                className="p-3 cursor-pointer hover:bg-daemon-primary/10 transition-colors"
                onClick={() => toggleSection('core')}
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-daemon-glow">Core Profile</h4>
                  <motion.svg
                    animate={{ rotate: expandedSections.has('core') ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                    className="w-4 h-4 text-slate-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </motion.svg>
                </div>
              </div>
              
              <AnimatePresence>
                {expandedSections.has('core') && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="p-3 pt-0 space-y-3"
                  >
                    {/* Trust & Attachment Metrics */}
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Trust Level</span>
                          <span className={`font-mono ${getMetricColor(coreModel.trust_level)}`}>
                            {formatMetric(coreModel.trust_level, '%')}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Distance</span>
                          <span className={`font-mono ${getMetricColor(coreModel.perceived_distance, true)}`}>
                            {formatMetric(coreModel.perceived_distance, '%')}
                          </span>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Anxiety</span>
                          <span className={`font-mono ${getMetricColor(coreModel.attachment_anxiety, true)}`}>
                            {formatMetric(coreModel.attachment_anxiety, '%')}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Energy</span>
                          <span className={`font-mono ${getMetricColor(coreModel.interaction_energy_level)}`}>
                            {formatMetric(coreModel.interaction_energy_level, '%')}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Processing Style */}
                    {coreModel.cognitive_processing_style && (
                      <div className="p-2 bg-daemon-primary/10 border border-daemon-accent/20 rounded">
                        <div className="text-xs text-slate-400 mb-1">Processing Style</div>
                        <div className="text-xs text-daemon-glow capitalize">
                          {coreModel.cognitive_processing_style}
                        </div>
                      </div>
                    )}

                    {/* Narrative Belief (if present) */}
                    {coreModel.narrative_belief && (
                      <div className="p-2 bg-obsidian-100/50 border border-daemon-accent/10 rounded">
                        <div className="text-xs text-slate-400 mb-1">Core Narrative</div>
                        <div className="text-xs text-slate-200 leading-relaxed max-h-32 overflow-y-auto">
                          {coreModel.narrative_belief}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Recent Insights */}
            {recentInsights.length > 0 && (
              <motion.div
                layout
                className="bg-obsidian-100/40 border border-daemon-accent/20 rounded-lg overflow-hidden"
              >
                <div
                  className="p-3 cursor-pointer hover:bg-daemon-primary/10 transition-colors"
                  onClick={() => toggleSection('insights')}
                >
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-daemon-glow">Recent Insights</h4>
                    <motion.svg
                      animate={{ rotate: expandedSections.has('insights') ? 180 : 0 }}
                      transition={{ duration: 0.2 }}
                      className="w-4 h-4 text-slate-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </motion.svg>
                  </div>
                </div>

                <AnimatePresence>
                  {expandedSections.has('insights') && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                      className="p-3 pt-0 space-y-2"
                    >
                      {recentInsights.map((insight, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="p-2 bg-obsidian-100/50 border border-daemon-accent/10 rounded"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-daemon-glow capitalize">
                              {insight.category}
                            </span>
                            <div className="flex items-center gap-2 text-[10px] text-slate-400">
                              <span>conf {insight.confidence.toFixed(2)}</span>
                              {insight.potential_misunderstanding ? (
                                <span className="text-orange-400">⚠</span>
                              ) : (
                                <span className="text-green-400">✓</span>
                              )}
                            </div>
                          </div>
                          <div className="text-xs text-slate-300 leading-relaxed">
                            {insight.description}
                          </div>
                        </motion.div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}

            {/* Active Personality Components */}
            {components.length > 0 && (
              <motion.div
                layout
                className="bg-obsidian-100/40 border border-daemon-accent/20 rounded-lg overflow-hidden"
              >
                <div
                  className="p-3 cursor-pointer hover:bg-daemon-primary/10 transition-colors"
                  onClick={() => toggleSection('components')}
                >
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-daemon-glow">Active Components</h4>
                    <motion.svg
                      animate={{ rotate: expandedSections.has('components') ? 180 : 0 }}
                      transition={{ duration: 0.2 }}
                      className="w-4 h-4 text-slate-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </motion.svg>
                  </div>
                </div>

                <AnimatePresence>
                  {expandedSections.has('components') && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                      className="p-3 pt-0 space-y-2"
                    >
                      {components.map((component, idx) => (
                        <motion.div
                          key={component.component_id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="p-2 bg-obsidian-100/50 border border-daemon-accent/10 rounded"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-daemon-glow">
                              {component.title}
                            </span>
                            <div className="text-[10px] text-slate-400 font-mono">
                              {formatMetric(component.confidence, '%')}
                            </div>
                          </div>
                          <div className="text-[11px] text-slate-400 mb-1 capitalize">
                            {component.category}
                          </div>
                          <div className="text-xs text-slate-300 leading-relaxed">
                            {component.description}
                          </div>
                        </motion.div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </>
        )}

        {!loading && !error && !coreModel && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center text-slate-500 py-6"
          >
            <div className="text-2xl mb-2">࿇</div>
            <p className="text-sm">No user model data available yet</p>
            <p className="text-xs text-slate-600 mt-1">Start a conversation to build the psyche profile</p>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default UserInsightsPanel;