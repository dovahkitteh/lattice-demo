/**
 * Dashboard Data Service Module
 * Handles all API communication and data fetching for the Daemon Dashboard
 */

class DashboardDataService {
    constructor(apiUrlBase) {
        this.apiUrlBase = apiUrlBase;
    }

    // ---------------------------------------------------------------------------
    // CORE STATUS FETCHING
    // ---------------------------------------------------------------------------

    async fetchDaemonStatus() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/status`);
            if (!response.ok) throw new Error('Failed to fetch daemon status');
            return await response.json();
        } catch (error) {
            console.error('Error fetching daemon status:', error);
            throw error;
        }
    }

    async fetchMemoryStats() {
        try {
            const response = await fetch(`${this.apiUrlBase}/memories/stats`);
            if (!response.ok) throw new Error('Failed to fetch memory stats');
            return await response.json();
        } catch (error) {
            console.error('Error fetching memory stats:', error);
            throw error;
        }
    }

    async fetchTokenUsage() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/token-usage`);
            if (!response.ok) throw new Error('Failed to fetch token usage');
            return await response.json();
        } catch (error) {
            console.error('Error fetching token usage:', error);
            // Return fallback data
            return {
                context_tokens: 0,
                max_context: 8192,
                usage_percentage: 0
            };
        }
    }

    async fetchHealthStatus() {
        try {
            const response = await fetch(`${this.apiUrlBase.replace('/v1', '')}/health`);
            if (!response.ok) throw new Error('Failed to fetch health status');
            return await response.json();
        } catch (error) {
            console.error('Error fetching health status:', error);
            throw error;
        }
    }

    // ---------------------------------------------------------------------------
    // DASHBOARD COMPONENT DATA
    // ---------------------------------------------------------------------------

    async fetchRecentMemories() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/recent-memories`);
            if (!response.ok) throw new Error('Failed to fetch recent memories');
            return await response.json();
        } catch (error) {
            console.error('Error fetching recent memories:', error);
            throw error;
        }
    }

    async fetchRecentEmotions() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/recent-emotions`);
            if (!response.ok) throw new Error('Failed to fetch recent emotions');
            return await response.json();
        } catch (error) {
            console.error('Error fetching recent emotions:', error);
            throw error;
        }
    }

    async fetchRecentPersonalityChanges() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/recent-personality`);
            if (!response.ok) throw new Error('Failed to fetch recent personality changes');
            return await response.json();
        } catch (error) {
            console.error('Error fetching recent personality changes:', error);
            throw error;
        }
    }

    // ---------------------------------------------------------------------------
    // DAEMON CONSCIOUSNESS DATA
    // ---------------------------------------------------------------------------

    async fetchDaemonThoughts() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/thoughts`);
            if (!response.ok) throw new Error('Failed to fetch daemon thoughts');
            return await response.json();
        } catch (error) {
            console.error('Error fetching daemon thoughts:', error);
            throw error;
        }
    }

    async fetchCurrentMood() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/mood/current`);
            if (!response.ok) throw new Error('Failed to fetch current mood');
            return await response.json();
        } catch (error) {
            console.error('Error fetching current mood:', error);
            throw error;
        }
    }

    async fetchUserAnalysis() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/user-model-detailed`);
            if (!response.ok) throw new Error('Failed to fetch user analysis');
            return await response.json();
        } catch (error) {
            console.error('Error fetching user analysis:', error);
            throw error;
        }
    }

    // ---------------------------------------------------------------------------
    // EMOTIONAL SYSTEM DATA
    // ---------------------------------------------------------------------------

    async fetchDetailedEmotionState() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/emotion-state`);
            if (!response.ok) throw new Error('Failed to fetch detailed emotion state');
            return await response.json();
        } catch (error) {
            console.error('Error fetching detailed emotion state:', error);
            throw error;
        }
    }

    async fetchActiveSeeds() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/active-seeds`);
            if (!response.ok) throw new Error('Failed to fetch active seeds');
            return await response.json();
        } catch (error) {
            console.error('Error fetching active seeds:', error);
            throw error;
        }
    }

    async fetchDistortionFrame() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/distortion-frame`);
            if (!response.ok) throw new Error('Failed to fetch distortion frame');
            return await response.json();
        } catch (error) {
            console.error('Error fetching distortion frame:', error);
            throw error;
        }
    }

    async fetchEmotionalMetrics() {
        try {
            const response = await fetch(`${this.apiUrlBase}/dashboard/emotional-metrics`);
            if (!response.ok) throw new Error('Failed to fetch emotional metrics');
            return await response.json();
        } catch (error) {
            console.error('Error fetching emotional metrics:', error);
            throw error;
        }
    }

    // ---------------------------------------------------------------------------
    // DEBUG DATA
    // ---------------------------------------------------------------------------

    async fetchTurnDebugInfo(sessionId) {
        try {
            const response = await fetch(`${this.apiUrlBase}/debug/turn_debug_info/${sessionId}`);
            if (!response.ok) throw new Error('Failed to fetch turn debug info');
            return await response.json();
        } catch (error) {
            console.error('Error fetching turn debug info:', error);
            throw error;
        }
    }

    async fetchMemoryInspectorData() {
        try {
            const response = await fetch(`${this.apiUrlBase}/debug/memory_inspector_data`);
            if (!response.ok) throw new Error('Failed to fetch memory inspector data');
            return await response.json();
        } catch (error) {
            console.error('Error fetching memory inspector data:', error);
            throw error;
        }
    }

    async fetchPersonalityTrackerData() {
        try {
            const response = await fetch(`${this.apiUrlBase}/debug/personality_tracker_data`);
            if (!response.ok) throw new Error('Failed to fetch personality tracker data');
            return await response.json();
        } catch (error) {
            console.error('Error fetching personality tracker data:', error);
            throw error;
        }
    }

    // ---------------------------------------------------------------------------
    // DAEMON SPECIFIC DATA
    // ---------------------------------------------------------------------------

    async fetchDaemonStatements() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/statements/recent`);
            if (!response.ok) throw new Error('Failed to fetch daemon statements');
            return await response.json();
        } catch (error) {
            console.error('Error fetching daemon statements:', error);
            throw error;
        }
    }

    async fetchShadowElements() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/shadow_elements`);
            if (!response.ok) throw new Error('Failed to fetch shadow elements');
            return await response.json();
        } catch (error) {
            console.error('Error fetching shadow elements:', error);
            throw error;
        }
    }

    async fetchRecursionBuffer() {
        try {
            const response = await fetch(`${this.apiUrlBase}/daemon/recursion/buffer`);
            if (!response.ok) throw new Error('Failed to fetch recursion buffer');
            return await response.json();
        } catch (error) {
            console.error('Error fetching recursion buffer:', error);
            throw error;
        }
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardDataService;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.DashboardDataService = DashboardDataService;
}