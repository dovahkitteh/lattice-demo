/**
 * Debug Tools Module
 * Handles debugging, inspection, and development utilities for the Daemon Dashboard
 */

class DebugTools {
    constructor(apiUrlBase, dataService, sessionManager) {
        this.apiUrlBase = apiUrlBase;
        this.dataService = dataService;
        this.sessionManager = sessionManager;
        this.currentTurnId = null;
        this.debugUpdateInterval = null;
        this.lastPersonalityUpdate = null;
        this.utils = DashboardUtils;
    }

    // ---------------------------------------------------------------------------
    // INITIALIZATION
    // ---------------------------------------------------------------------------

    initializeDebugComponents() {
        this.currentTurnId = null;
        this.debugUpdateInterval = null;
        
        // Setup debug event listeners
        this.setupDebugEventListeners();
        
        // Setup periodic debug updates
        this.setupDebugIntervals();
        
        console.log("Debug components initialized.");
    }

    setupDebugEventListeners() {
        // Refresh debug button
        const refreshDebugBtn = document.getElementById('refreshDebugBtn');
        if (refreshDebugBtn) {
            refreshDebugBtn.onclick = () => this.refreshTurnDebugInfo();
        }
        
        // Export debug button
        const exportDebugBtn = document.getElementById('exportDebugBtn');
        if (exportDebugBtn) {
            exportDebugBtn.onclick = () => this.exportDebugData();
        }
        
        // Memory inspector controls
        const refreshMemoryBtn = document.getElementById('refreshMemoryBtn');
        if (refreshMemoryBtn) {
            refreshMemoryBtn.onclick = () => this.refreshMemoryInspector();
        }
        
        const memoryFilter = document.getElementById('memoryFilter');
        if (memoryFilter) {
            memoryFilter.onchange = (e) => this.filterMemoryInspector(e.target.value);
        }
        
        // Force statement buttons
        const forceStatementBtn = document.getElementById('forceStatementBtn');
        if (forceStatementBtn) {
            forceStatementBtn.onclick = () => this.forceStatement();
        }

        const forceIntegrationBtn = document.getElementById('forceIntegrationBtn');
        if (forceIntegrationBtn) {
            forceIntegrationBtn.onclick = () => this.forceShadowIntegration();
        }

        const resetBufferBtn = document.getElementById('resetBufferBtn');
        if (resetBufferBtn) {
            resetBufferBtn.onclick = () => this.resetRecursionBuffer();
        }
        
        console.log("Debug event listeners setup complete.");
    }

    setupDebugIntervals() {
        // Refresh debug info every 30 seconds
        this.debugUpdateInterval = setInterval(() => {
            this.refreshTurnDebugInfo();
            this.refreshMemoryInspector();
            // Refresh personality tracker less frequently (every 60 seconds)
            if (!this.lastPersonalityUpdate || Date.now() - this.lastPersonalityUpdate > 60000) {
                this.refreshPersonalityTracker();
                this.lastPersonalityUpdate = Date.now();
            }
        }, 30000);
        
        console.log("Debug intervals setup complete.");
    }

    cleanup() {
        if (this.debugUpdateInterval) {
            clearInterval(this.debugUpdateInterval);
            this.debugUpdateInterval = null;
        }
    }

    // ---------------------------------------------------------------------------
    // TURN DEBUG INFORMATION
    // ---------------------------------------------------------------------------

    async refreshTurnDebugInfo() {
        const activeSessionId = this.sessionManager.getActiveSessionId();
        if (!activeSessionId) {
            return;
        }
        
        try {
            const data = await this.dataService.fetchTurnDebugInfo(activeSessionId);
            this.updateTurnDebugUI(data);
        } catch (error) {
            console.error('Error refreshing turn debug info:', error);
        }
    }

    updateTurnDebugUI(data) {
        let formatted;
        
        if (data.formatted_data) {
            // New format with turn analyzer
            formatted = data.formatted_data;
        } else if (data.memory_info || data.personality_changes || data.processing_stats) {
            // Legacy fallback format
            formatted = this.createLegacyFormattedData(data);
        } else {
            console.warn('No debug data available');
            return;
        }

        // Update memory info section
        this.updateDebugSection('memoryInfoContent', this.formatMemoryInfo(formatted.memory_info || {}));
        
        // Update emotion info section
        this.updateDebugSection('emotionInfoContent', this.formatEmotionInfo(formatted.emotion_info || {}));
        
        // Update personality info section
        this.updateDebugSection('personalityInfoContent', this.formatPersonalityInfo(formatted.personality_info || {}));
        
        // Update processing stats section
        this.updateDebugSection('processingStatsContent', this.formatProcessingStats(formatted.processing_stats || {}));
    }

    createLegacyFormattedData(data) {
        return {
            memory_info: {
                memories_stored: data.memory_info?.memories_stored || 0,
                dual_channel_active: data.memory_info?.storage_summary?.has_dual_channel || false,
                total_affect_magnitude: data.memory_info?.storage_summary?.total_affect_magnitude || 0,
                explanation: `Stored ${data.memory_info?.memories_stored || 0} memories with ${data.memory_info?.storage_summary?.has_reflections ? 'reflections' : 'no reflections'}`
            },
            emotion_info: {
                dominant_user_emotions: [],
                dominant_self_emotions: [],
                emotional_influence: 0,
                explanation: "Legacy emotion data - limited information available"
            },
            personality_info: {
                user_model_updates: data.personality_changes?.user_model_changes || {},
                shadow_changes: data.personality_changes?.shadow_changes || {},
                daemon_changes: data.personality_changes?.personality_changes || {},
                explanation: "Personality system evolution in progress"
            },
            processing_stats: data.processing_stats || {}
        };
    }

    updateDebugSection(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    }

    formatMemoryInfo(memoryInfo) {
        return `
            <div class="debug-info-section">
                <h6>Memory Storage</h6>
                <div class="debug-metrics">
                    <div class="debug-metric">
                        <span class="metric-label">Memories Stored:</span>
                        <span class="metric-value">${memoryInfo.memories_stored || 0}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Dual Channel:</span>
                        <span class="metric-value">${memoryInfo.dual_channel_active ? 'Active' : 'Inactive'}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Total Affect:</span>
                        <span class="metric-value">${(memoryInfo.total_affect_magnitude || 0).toFixed(3)}</span>
                    </div>
                </div>
                <div class="debug-explanation">
                    ${memoryInfo.explanation || 'No memory information available'}
                </div>
            </div>
        `;
    }

    formatEmotionInfo(emotionInfo) {
        const userEmotions = Array.isArray(emotionInfo.dominant_user_emotions) ? 
            emotionInfo.dominant_user_emotions.join(', ') : 'none detected';
        const selfEmotions = Array.isArray(emotionInfo.dominant_self_emotions) ? 
            emotionInfo.dominant_self_emotions.join(', ') : 'none detected';

        return `
            <div class="debug-info-section">
                <h6>Emotional Analysis</h6>
                <div class="debug-metrics">
                    <div class="debug-metric">
                        <span class="metric-label">User Emotions:</span>
                        <span class="metric-value">${userEmotions}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">AI Emotions:</span>
                        <span class="metric-value">${selfEmotions}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Influence Score:</span>
                        <span class="metric-value">${(emotionInfo.emotional_influence || 0).toFixed(3)}</span>
                    </div>
                </div>
                <div class="debug-explanation">
                    ${emotionInfo.explanation || 'No emotional analysis available'}
                </div>
            </div>
        `;
    }

    formatPersonalityInfo(personalityInfo) {
        const userModelUpdates = Object.keys(personalityInfo.user_model_updates || {}).length;
        const shadowChanges = Object.keys(personalityInfo.shadow_changes || {}).length;
        const daemonChanges = Object.keys(personalityInfo.daemon_changes || {}).length;

        return `
            <div class="debug-info-section">
                <h6>Personality Evolution</h6>
                <div class="debug-metrics">
                    <div class="debug-metric">
                        <span class="metric-label">User Model Updates:</span>
                        <span class="metric-value">${userModelUpdates}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Shadow Changes:</span>
                        <span class="metric-value">${shadowChanges}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Daemon Changes:</span>
                        <span class="metric-value">${daemonChanges}</span>
                    </div>
                </div>
                <div class="debug-explanation">
                    ${personalityInfo.explanation || 'Personality system monitoring active'}
                </div>
            </div>
        `;
    }

    formatProcessingStats(processingStats) {
        return `
            <div class="debug-info-section">
                <h6>Processing Statistics</h6>
                <div class="debug-metrics">
                    <div class="debug-metric">
                        <span class="metric-label">Response Time:</span>
                        <span class="metric-value">${processingStats.response_time || 'N/A'}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Tokens Used:</span>
                        <span class="metric-value">${processingStats.tokens_used || 'N/A'}</span>
                    </div>
                    <div class="debug-metric">
                        <span class="metric-label">Model:</span>
                        <span class="metric-value">${processingStats.model || 'N/A'}</span>
                    </div>
                </div>
            </div>
        `;
    }

    // ---------------------------------------------------------------------------
    // MEMORY INSPECTOR
    // ---------------------------------------------------------------------------

    async refreshMemoryInspector() {
        try {
            const data = await this.dataService.fetchMemoryInspectorData();
            this.updateMemoryInspectorUI(data);
        } catch (error) {
            console.error('Error refreshing memory inspector:', error);
        }
    }

    updateMemoryInspectorUI(data) {
        if (!data) return;

        const summary = data.summary || {};
        const memories = data.memories || [];

        // Update summary statistics
        this.updateElement('totalMemories', summary.total_count || 0);
        this.updateElement('recentMemories', summary.recent_count || 0);
        this.updateElement('averageAffect', (summary.avg_affect || 0).toFixed(3));
        this.updateElement('dualChannelCount', summary.dual_channel_count || 0);

        // Update memory list
        const memoryList = document.getElementById('memoryList');
        if (memoryList && memories.length > 0) {
            memoryList.innerHTML = '';
            
            memories.slice(0, 20).forEach(memory => { // Limit to 20 for performance
                const memoryItem = this.createMemoryItem(memory);
                memoryList.appendChild(memoryItem);
            });
        }
    }

    createMemoryItem(memory) {
        const item = document.createElement('div');
        item.className = 'memory-inspector-item';
        
        const userAffect = memory.user_affect_magnitude || 0;
        const selfAffect = memory.self_affect_magnitude || 0;
        const significance = memory.emotional_significance || 0;
        
        item.innerHTML = `
            <div class="memory-item-header">
                <span class="memory-id">ID: ${memory.id.substring(0, 8)}...</span>
                <span class="memory-timestamp">${this.utils.formatTimestamp(memory.created_at)}</span>
            </div>
            <div class="memory-content">${memory.title || 'No title'}</div>
            <div class="memory-affects">
                <span class="affect-metric">User: ${userAffect.toFixed(2)}</span>
                <span class="affect-metric">AI: ${selfAffect.toFixed(2)}</span>
                <span class="affect-metric">Significance: ${significance.toFixed(2)}</span>
            </div>
        `;
        
        return item;
    }

    filterMemoryInspector(filterValue) {
        const memoryItems = document.querySelectorAll('.memory-inspector-item');
        
        memoryItems.forEach(item => {
            const content = item.textContent.toLowerCase();
            const shouldShow = filterValue === 'all' || 
                             (filterValue === 'high-affect' && content.includes('significance')) ||
                             (filterValue === 'recent' && true); // Could implement actual recent filtering
            
            item.style.display = shouldShow ? 'block' : 'none';
        });
    }

    // ---------------------------------------------------------------------------
    // PERSONALITY TRACKER
    // ---------------------------------------------------------------------------

    async refreshPersonalityTracker() {
        try {
            const data = await this.dataService.fetchPersonalityTrackerData();
            this.updatePersonalityTrackerUI(data);
        } catch (error) {
            console.error('Error refreshing personality tracker:', error);
        }
    }

    updatePersonalityTrackerUI(data) {
        if (!data) return;

        const userModelState = data.user_model_state || {};
        const shadowState = data.shadow_state || {};
        const personalityState = data.personality_state || {};

        // Update user model section
        this.updateElement('trustLevel', ((userModelState.trust_level || 0.7) * 100).toFixed(0) + '%');
        this.updateElement('attachmentAnxiety', ((userModelState.attachment_anxiety || 0.2) * 100).toFixed(0) + '%');
        this.updateElement('perceivedDistance', ((userModelState.perceived_distance || 0.3) * 100).toFixed(0) + '%');

        // Update shadow integration section  
        this.updateElement('shadowCharge', (shadowState.integration_charge || 0).toFixed(2));
        this.updateElement('shadowElements', shadowState.active_elements || 0);
        this.updateElement('shadowPressure', (shadowState.integration_pressure || 0).toFixed(2));

        // Update personality evolution section
        this.updateElement('evolutionRate', (personalityState.evolution_rate || 0).toFixed(3));
        this.updateElement('stabilityIndex', (personalityState.stability_index || 0.8).toFixed(2));
        this.updateElement('adaptationScore', (personalityState.adaptation_score || 0.75).toFixed(2));
    }

    // ---------------------------------------------------------------------------
    // CONTROL FUNCTIONS
    // ---------------------------------------------------------------------------

    async forceStatement() {
        try {
            console.log('Forcing daemon statement generation...');
            const response = await fetch(`${this.apiUrlBase}/daemon/force_statement`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Statement forced:', data);
                await this.loadDaemonStatements();
            } else {
                console.error('Failed to force statement');
            }
        } catch (error) {
            console.error('Error forcing statement:', error);
        }
    }

    async forceShadowIntegration() {
        try {
            console.log('Forcing shadow integration...');
            const response = await fetch(`${this.apiUrlBase}/daemon/force_shadow_integration`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                console.log('Shadow integration forced');
                await this.loadShadowElements();
            } else {
                console.error('Failed to force shadow integration');
            }
        } catch (error) {
            console.error('Error forcing shadow integration:', error);
        }
    }

    async resetRecursionBuffer() {
        if (!confirm('Reset recursion buffer? This will clear all buffered experiences.')) {
            return;
        }

        try {
            console.log('Resetting recursion buffer...');
            const response = await fetch(`${this.apiUrlBase}/daemon/reset_buffer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                console.log('Recursion buffer reset');
                await this.loadRecursionBuffer();
            } else {
                console.error('Failed to reset recursion buffer');
            }
        } catch (error) {
            console.error('Error resetting recursion buffer:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // DATA LOADING HELPERS
    // ---------------------------------------------------------------------------

    async loadDaemonStatements() {
        try {
            const data = await this.dataService.fetchDaemonStatements();
            // Update statements display if element exists
            const container = document.getElementById('daemonStatementsContent');
            if (container && data.statements) {
                container.innerHTML = data.statements.slice(0, 5).map(statement => 
                    `<div class="statement-item">${statement.content}</div>`
                ).join('');
            }
        } catch (error) {
            console.error('Error loading daemon statements:', error);
        }
    }

    async loadShadowElements() {
        try {
            const data = await this.dataService.fetchShadowElements();
            // Update shadow elements display if element exists
            const container = document.getElementById('shadowElementsContent');
            if (container && data.elements) {
                container.innerHTML = data.elements.slice(0, 5).map(element => 
                    `<div class="shadow-element">${element.description}</div>`
                ).join('');
            }
        } catch (error) {
            console.error('Error loading shadow elements:', error);
        }
    }

    async loadRecursionBuffer() {
        try {
            const data = await this.dataService.fetchRecursionBuffer();
            // Update recursion buffer display if element exists
            const container = document.getElementById('recursionBufferContent');
            if (container && data.buffer_contents) {
                container.innerHTML = data.buffer_contents.slice(0, 10).map(item => 
                    `<div class="buffer-item">${item.summary || item.description}</div>`
                ).join('');
            }
        } catch (error) {
            console.error('Error loading recursion buffer:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // DATA EXPORT
    // ---------------------------------------------------------------------------

    async exportDebugData() {
        try {
            const activeSessionId = this.sessionManager.getActiveSessionId();
            if (!activeSessionId) {
                alert('No active session to export debug data for');
                return;
            }

            // Gather all debug data
            const debugData = {
                session_id: activeSessionId,
                timestamp: new Date().toISOString(),
                turn_debug: await this.dataService.fetchTurnDebugInfo(activeSessionId),
                memory_inspector: await this.dataService.fetchMemoryInspectorData(),
                personality_tracker: await this.dataService.fetchPersonalityTrackerData()
            };

            // Create and download file
            const blob = new Blob([JSON.stringify(debugData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `daemon_debug_${activeSessionId.substring(0, 8)}_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);

            console.log('Debug data exported successfully');
        } catch (error) {
            console.error('Error exporting debug data:', error);
            alert(`Failed to export debug data: ${error.message}`);
        }
    }

    // ---------------------------------------------------------------------------
    // UTILITY METHODS
    // ---------------------------------------------------------------------------

    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            const oldValue = element.textContent;
            element.textContent = value !== null && value !== undefined ? value.toString() : 'N/A';
            
            // Add visual feedback for changes
            if (oldValue !== element.textContent) {
                element.style.backgroundColor = 'var(--highlight-color)';
                setTimeout(() => {
                    element.style.backgroundColor = '';
                }, 1000);
            }
        }
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DebugTools;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.DebugTools = DebugTools;
}