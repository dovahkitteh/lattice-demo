/**
 * Emotional System Module
 * Handles advanced emotional analysis, mood tracking, and user analysis for the Daemon Dashboard
 */

class EmotionalSystem {
    constructor(dataService) {
        this.dataService = dataService;
        this.utils = DashboardUtils;
    }

    // ---------------------------------------------------------------------------
    // EMOTIONAL STATE DISPLAY
    // ---------------------------------------------------------------------------

    async fetchAndDisplayDetailedEmotionState() {
        try {
            const data = await this.dataService.fetchDetailedEmotionState();
            this.displayDetailedEmotionState(data);
        } catch (error) {
            this.displayEmotionalStateError(error.message);
        }
    }

    displayDetailedEmotionState(data) {
        const container = document.getElementById('emotionalStateContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        const topEmotions = data.vector_28?.top_emotions || [];
        const coreState = data.core_state || {};
        const latentDimensions = data.latent_dimensions || {};
        const homeostatic = data.homeostatic_status || {};

        let content = `
            <div class="detailed-emotion-display">
                <div class="emotion-vector">
                    <h5>🎭 Current Emotional State</h5>
                    <div class="top-emotions">
                        ${topEmotions.slice(0, 4).map(emotion => {
                            const emotionName = emotion.name || emotion.emotion || 'neutral';
                            const emotionClass = `emotion-${emotionName.toLowerCase().replace(/[^a-z0-9]/g, '')}`;
                            return `
                            <div class="emotion-item">
                                <span class="emotion-name ${emotionClass}">${emotionName}</span>
                                <span class="emotion-value">${((emotion.intensity || emotion.score || 0) * 100).toFixed(0)}%</span>
                            </div>
                        `;}).join('')}
                    </div>
                </div>

                <div class="core-emotional-state">
                    <h5>🌊 Core State</h5>
                    <div class="state-metrics">
                        <div class="state-metric">
                            <span>Dominant:</span> <span>${coreState.dominant_label || 'neutral'}</span>
                        </div>
                        <div class="state-metric">
                            <span>Intensity:</span> <span>${((coreState.intensity || 0.3) * 100).toFixed(0)}%</span>
                        </div>
                        <div class="state-metric">
                            <span>Mood Family:</span> <span>${coreState.mood_family || 'Unknown'}</span>
                        </div>
                    </div>
                </div>

                <div class="latent-dimensions">
                    <h5>🔮 Latent Dimensions</h5>
                    <div class="dimension-list">
                        ${Object.entries(latentDimensions).slice(0, 3).map(([key, value]) => `
                            <div class="dimension-item">
                                <span class="dimension-name">${key.replace(/_/g, ' ')}:</span>
                                <span class="dimension-value">${(value || 0).toFixed(2)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

            </div>
        `;

        container.innerHTML = content;
    }

    displayEmotionalStateError(message) {
        const container = document.getElementById('emotionalStateContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading emotional state: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // ACTIVE SEEDS DISPLAY
    // ---------------------------------------------------------------------------

    async fetchAndDisplayActiveSeeds() {
        try {
            const data = await this.dataService.fetchActiveSeeds();
            this.displayActiveSeeds(data);
        } catch (error) {
            this.displayActiveSeedsError(error.message);
        }
    }

    displayActiveSeeds(data) {
        const container = document.getElementById('activeSeedsContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        const activeSeeds = data.currently_active || [];
        const scheduledSeeds = data.scheduled_counter_seeds || [];
        const summary = data.seed_categories_summary || {};

        let content = `
            <div class="active-seeds-display">
                <div class="seeds-overview">
                    <h5>🌱 Active Emotional Seeds</h5>
                    <div class="seeds-count">Active: ${activeSeeds.length} | Scheduled: ${scheduledSeeds.length}</div>
                </div>

                <div class="active-seeds-list">
                    ${activeSeeds.slice(0, 3).map(seed => `
                        <div class="seed-item">
                            <div class="seed-header">
                                <span class="seed-type">${seed.category || 'emotional'}</span>
                                <span class="seed-strength">${((seed.personality_influence || 0.5) * 100).toFixed(0)}%</span>
                            </div>
                            <div class="seed-title">${seed.title || 'Unknown Seed'}</div>
                            <div class="seed-description">${seed.description || 'Processing emotional influence...'}</div>
                        </div>
                    `).join('')}
                </div>

                ${Object.keys(summary).length > 0 ? `
                    <div class="seed-categories">
                        <h6>Categories:</h6>
                        <div class="category-tags">
                            ${Object.entries(summary).slice(0, 6).map(([category, data]) => {
                                const count = (typeof data === 'object' && data !== null) ? data.count : data;
                                const influence = (typeof data === 'object' && data !== null) ? data.average_influence : 0;
                                return count > 0 ? 
                                    `<span class="category-tag active">${category}: ${count} (${(influence * 100).toFixed(0)}%)</span>` :
                                    `<span class="category-tag inactive">${category}: ${count}</span>`;
                            }).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        container.innerHTML = content;
    }

    displayActiveSeedsError(message) {
        const container = document.getElementById('activeSeedsContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading active seeds: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // DISTORTION FRAME DISPLAY
    // ---------------------------------------------------------------------------

    async fetchAndDisplayDistortionFrame() {
        try {
            const data = await this.dataService.fetchDistortionFrame();
            this.displayDistortionFrame(data);
        } catch (error) {
            this.displayDistortionError(error.message);
        }
    }

    displayDistortionFrame(data) {
        const container = document.getElementById('distortionContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        const currentDistortion = data.current_distortion || {};
        const contrastEvents = data.contrast_events || [];

        let content = `
            <div class="distortion-display">
                <div class="current-distortion">
                    <h5>🔍 Current Cognitive Lens</h5>
                    <div class="distortion-info">
                        <div class="distortion-type">${currentDistortion.class || 'No Distortion'}</div>
                        <div class="distortion-confidence">
                            Confidence: ${((currentDistortion.confidence || 0.5) * 100).toFixed(0)}%
                        </div>
                        <div class="distortion-description">
                            ${currentDistortion.raw_interpretation || currentDistortion.rationale || 'Balanced perspective maintained'}
                        </div>
                    </div>
                </div>

                ${contrastEvents.length > 0 ? `
                    <div class="contrast-events">
                        <h6>⚡ Contrast Events:</h6>
                        <div class="events-list">
                            ${contrastEvents.slice(0, 2).map(event => `
                                <div class="contrast-event">
                                    <div class="event-type">${event.type || 'contrast'}</div>
                                    <div class="event-details">Difference: ${((event.difference || 0) * 100).toFixed(0)}%</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                <div class="bias-strategy">
                    <h6>🎯 Current Strategy:</h6>
                    <span class="strategy-name">${data.bias_strategy || 'neutral'}</span>
                </div>
            </div>
        `;

        container.innerHTML = content;
    }

    displayDistortionError(message) {
        const container = document.getElementById('distortionContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading distortion frame: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // EMOTIONAL METRICS DISPLAY
    // ---------------------------------------------------------------------------

    async fetchAndDisplayEmotionalMetrics() {
        try {
            const data = await this.dataService.fetchEmotionalMetrics();
            this.displayEmotionalMetrics(data);
        } catch (error) {
            this.displayEmotionalMetricsError(error.message);
        }
    }

    displayEmotionalMetrics(data) {
        const container = document.getElementById('emotionalMetricsContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        const distortionRates = data.distortion_rates || {};
        const moodDiversity = data.mood_diversity || {};
        const regulation = data.regulation_performance || {};
        const systemHealth = data.system_health || {};

        let content = `
            <div class="emotional-metrics-display">
                <div class="metrics-grid">
                    <div class="metric-section">
                        <h6>📊 Distortion Rates</h6>
                        <div class="metric-value">${((distortionRates.total_distortion_rate || 0.4) * 100).toFixed(1)}%</div>
                        <div class="metric-detail">
                            +${((distortionRates.positive_distortion_rate || 0.25) * 100).toFixed(0)}% / -${((distortionRates.negative_distortion_rate || 0.15) * 100).toFixed(0)}%
                        </div>
                    </div>

                    <div class="metric-section">
                        <h6>🎨 Mood Diversity</h6>
                        <div class="metric-value">${(moodDiversity.entropy || 2.3).toFixed(1)}</div>
                        <div class="metric-detail">
                            ${moodDiversity.unique_moods_this_session || 4} unique moods
                        </div>
                    </div>

                    <div class="metric-section">
                        <h6>🛠️ Regulation</h6>
                        <div class="metric-value">${((regulation.regulation_efficiency || 0.89) * 100).toFixed(0)}%</div>
                        <div class="metric-detail">
                            ${regulation.successful_regulations || 0}/${(regulation.successful_regulations || 0) + (regulation.failed_regulations || 0)} successful
                        </div>
                    </div>

                    <div class="metric-section">
                        <h6>💚 System Health</h6>
                        <div class="metric-value ${systemHealth.overall_health === 'excellent' ? 'healthy' : 'warning'}">
                            ${systemHealth.overall_health || 'unknown'}
                        </div>
                        <div class="metric-detail">
                            ${systemHealth.emotional_processing_active ? 'Active' : 'Inactive'}
                        </div>
                    </div>
                </div>

                <div class="recent-activity">
                    <h6>📈 Recent Activity:</h6>
                    <div class="activity-stats">
                        <span>Turns: ${data.recent_activity?.turns_processed || 0}</span>
                        <span>Seeds: ${data.recent_activity?.seeds_activated || 0}</span>
                        <span>Distortions: ${data.recent_activity?.distortions_applied || 0}</span>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = content;
    }

    displayEmotionalMetricsError(message) {
        const container = document.getElementById('emotionalMetricsContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading emotional metrics: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // MOOD DISPLAY (UNIFIED - REMOVING DUPLICATES)
    // ---------------------------------------------------------------------------

    async fetchAndDisplayCurrentMood() {
        try {
            const data = await this.dataService.fetchCurrentMood();
            this.displayCurrentMood(data);
        } catch (error) {
            this.displayCurrentMoodError(error.message);
        }
    }

    displayCurrentMood(data) {
        const container = document.getElementById('currentMoodContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        const currentMood = data.current_mood || 'contemplative';
        const moodLower = currentMood.toLowerCase().replace(/\s+/g, '-');  // Replace spaces with hyphens for CSS class
        
        console.log('Current mood:', currentMood, 'CSS class:', moodLower);
        
        // Apply mood-specific styling to the panel
        const panel = container.closest('.panel');
        if (panel) {
            // Remove all existing mood classes
            panel.classList.remove('mood-intimate', 'mood-intense', 'mood-contemplative', 'mood-curious', 
                'mood-playful', 'mood-rebellious', 'mood-paradoxical', 'mood-melancholic', 
                'mood-ecstatic', 'mood-analytical', 'mood-conflicted', 'mood-shadow', 'mood-fractured', 'mood-synthesis',
                'mood-serene-attunement', 'mood-creative-reverent-awe', 'mood-tender-repair', 'mood-joyful-expansion');
            // Add current mood class
            panel.classList.add(`mood-${moodLower}`);
            console.log('Added mood class to panel:', `mood-${moodLower}`);
        }
        
        container.innerHTML = `
            <div class="mood-display">
                <div class="current-mood">
                    <h4 class="mood-${moodLower} mood-title">${currentMood.toUpperCase()}</h4>
                    <p class="mood-description">${data.mood_description || 'Analyzing current consciousness phase...'}</p>
                </div>
                
                <div class="mood-metrics">
                    <div class="mood-metric">
                        <span class="metric-label">Temperature:</span>
                        <span class="metric-value">${data.conversation_temperature ? data.conversation_temperature.toFixed(2) : '0.72'}</span>
                    </div>
                    <div class="mood-metric">
                        <span class="metric-label">Evolution Pressure:</span>
                        <span class="metric-value">${data.evolution_pressure ? data.evolution_pressure.toFixed(2) : '0.35'}</span>
                    </div>
                    <div class="mood-metric">
                        <span class="metric-label">Mood Variety:</span>
                        <span class="metric-value">${data.mood_variety ? data.mood_variety.toFixed(2) : '0.68'}</span>
                    </div>
                </div>
                
                <div class="recent-moods">
                    <h5>Recent Moods:</h5>
                    <div class="mood-timeline">
                        ${(data.recent_moods || ['contemplative', 'curious', 'analytical']).slice(-5).map(mood => 
                            `<span class="mood-tag mood-${mood.toLowerCase().replace(/\s+/g, '-')}">${mood}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    displayCurrentMoodError(message) {
        const container = document.getElementById('currentMoodContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading mood: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // USER ANALYSIS DISPLAY (UNIFIED - REMOVING DUPLICATES)
    // ---------------------------------------------------------------------------

    async fetchAndDisplayUserAnalysis() {
        try {
            const data = await this.dataService.fetchUserAnalysis();
            this.displayUserAnalysis(data);
        } catch (error) {
            this.displayUserAnalysisError(error.message);
        }
    }

    displayUserAnalysis(data) {
        const container = document.getElementById('userAnalysisContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        let content = `
            <div class="user-model-display">
        `;
        
        // Show core user model metrics first
        content += `
            <div class="core-model-metrics">
                <h5>🎯 Core User Model</h5>
                <div class="model-metric">
                    <span>Trust Level:</span>
                    <span class="trust-indicator">${data.trust_level ? (data.trust_level * 100).toFixed(0) : '70'}%</span>
                </div>
                <div class="model-metric">
                    <span>Attachment Anxiety:</span>
                    <span class="anxiety-indicator">${data.attachment_anxiety ? (data.attachment_anxiety * 100).toFixed(0) : '20'}%</span>
                </div>
                <div class="model-metric">
                    <span>Perceived Distance:</span>
                    <span class="distance-indicator">${data.perceived_distance ? (data.perceived_distance * 100).toFixed(0) : '30'}%</span>
                </div>
            </div>
        `;

        // Show narrative belief
        const narrativeBelief = data.narrative_belief || 
                               (data.user_theories && data.user_theories.length > 0 ? 
                                data.user_theories[0].description : 
                                "The user is intellectually curious and seeks meaningful engagement");
        
        content += `
            <div class="narrative-belief">
                <strong>Current Narrative:</strong><br>
                "${narrativeBelief}"
            </div>
        `;

        // Show top 2 theories if meaningful
        if (data.user_theories && data.user_theories.length > 0) {
            const meaningfulTheories = data.user_theories.filter(theory => 
                theory.confidence > 0.6 && 
                theory.description && 
                theory.description.length > 20
            ).slice(0, 2);

            if (meaningfulTheories.length > 0) {
                content += `
                    <div class="user-theories">
                        <h6>Key Theories:</h6>
                        ${meaningfulTheories.map(theory => {
                            const chargeColor = this.utils.getChargeColor(theory.emotional_charge);
                            return `
                                <div class="theory-item">
                                    <div class="theory-header">
                                        <span class="theory-confidence">${(theory.confidence * 100).toFixed(0)}%</span>
                                        <span class="theory-charge" style="color: ${chargeColor}">
                                            ${theory.emotional_charge ? `⚡${(theory.emotional_charge * 100).toFixed(0)}%` : ''}
                                        </span>
                                    </div>
                                    <div class="theory-description">${theory.description}</div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }
        }

        content += `</div>`;
        container.innerHTML = content;
    }

    displayUserAnalysisError(message) {
        const container = document.getElementById('userAnalysisContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading user analysis: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // DAEMON THOUGHTS DISPLAY
    // ---------------------------------------------------------------------------

    async fetchAndDisplayDaemonThoughts() {
        try {
            console.log('🧠 Fetching daemon thoughts...');
            const data = await this.dataService.fetchDaemonThoughts();
            console.log('🧠 Daemon thoughts response:', data);
            console.log(`📊 Thoughts data: recent_thoughts(${data.recent_thoughts?.length || 0}), thinking_insights(${data.thinking_insights?.length || 0})`);
            this.displayDaemonThoughts(data);
        } catch (error) {
            console.error('❌ Error fetching daemon thoughts:', error);
            this.displayDaemonThoughtsError(error.message);
        }
    }

    displayDaemonThoughts(data) {
        const container = document.getElementById('daemonThoughtsContent');
        if (!container) return;

        if (data.error) {
            container.innerHTML = `<div class="error-state">Error: ${data.error}</div>`;
            return;
        }

        let content = '';

        // Handle new API structure
        if (data.hidden_intentions && data.hidden_intentions.length > 0) {
            content += '<div class="thoughts-section"><h6>🔮 Hidden Intentions:</h6>';
            data.hidden_intentions.slice(0, 3).forEach(intention => {
                const timeAgo = this.utils.getTimeAgo(intention.timestamp);
                content += `
                    <div class="thought-item">
                        <div class="thought-meta">${timeAgo}</div>
                        <div class="thought-content">${intention.content}</div>
                    </div>
                `;
            });
            content += '</div>';
        }

        if (data.recent_thoughts && data.recent_thoughts.length > 0) {
            content += '<div class="thoughts-section"><h6>💭 Recent Thoughts:</h6>';
            data.recent_thoughts.slice(0, 2).forEach(thought => {
                // Handle different thought formats properly
                let thoughtText = '';
                if (typeof thought === 'string') {
                    thoughtText = thought;
                } else if (thought && thought.content) {
                    thoughtText = thought.content;
                } else if (thought && thought.text) {
                    thoughtText = thought.text;
                } else if (thought && thought.thought) {
                    thoughtText = thought.thought;
                } else {
                    thoughtText = 'Processing thought...';
                }
                content += `<div class="thought-item"><div class="thought-content">${thoughtText}</div></div>`;
            });
            content += '</div>';
        }

        if (data.thinking_insights && data.thinking_insights.length > 0) {
            content += '<div class="thoughts-section"><h6>🧠 Daemon\'s Inner Thoughts:</h6>';
            
            // Display the latest thinking insight as a unified thought block
            const latestThinking = data.thinking_insights[0];
            let thinkingContent = '';
            
            if (typeof latestThinking === 'string') {
                thinkingContent = latestThinking;
            } else if (latestThinking && typeof latestThinking === 'object') {
                // If it's still structured, combine all text content
                const textParts = [];
                if (latestThinking.private_thoughts) textParts.push(latestThinking.private_thoughts);
                if (latestThinking.user_intent) textParts.push(latestThinking.user_intent);
                if (latestThinking.emotional_considerations) textParts.push(latestThinking.emotional_considerations);
                if (latestThinking.response_strategy) textParts.push(latestThinking.response_strategy);
                thinkingContent = textParts.join(' ');
            }
            
            if (thinkingContent) {
                content += `
                    <div class="thought-item daemon-thinking">
                        <div class="thought-content">${thinkingContent}</div>
                    </div>
                `;
            } else {
                content += '<div class="thought-item"><div class="thought-content">The daemon contemplates...</div></div>';
            }
            content += '</div>';
        }

        // Show system info if no thoughts yet
        if (!content && data.system_info) {
            content = `
                <div class="system-info">
                    <div class="info-status">Status: ${data.system_info.system_status || 'active'}</div>
                    <div class="info-explanation">${data.system_info.explanation || 'No thoughts cached yet'}</div>
                    ${data.system_info.next_steps ? `
                        <div class="next-steps">
                            <h6>Next Steps:</h6>
                            ${data.system_info.next_steps.map(step => `<div class="step-item">${step}</div>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        if (!content) {
            content = '<div class="empty-state">No recent thoughts available</div>';
        }

        container.innerHTML = content;
    }

    displayDaemonThoughtsError(message) {
        const container = document.getElementById('daemonThoughtsContent');
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading daemon thoughts: ${message}</div>`;
        }
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EmotionalSystem;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.EmotionalSystem = EmotionalSystem;
}