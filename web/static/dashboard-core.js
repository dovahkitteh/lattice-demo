/**
 * Dashboard Core Module
 * Main orchestration and initialization for the Daemon Dashboard
 */

class DashboardCore {
    constructor() {
        this.apiUrlBase = '/v1';
        this.charts = {};
        this.updateIntervals = {};
        
        // Initialize service modules
        this.dataService = new DashboardDataService(this.apiUrlBase);
        this.sessionManager = new SessionManager(`${this.apiUrlBase}/conversations/sessions`, this.dataService);
        this.dashboardComponents = new DashboardComponents(this.dataService);
        this.chatInterface = new ChatInterface(this.apiUrlBase, this.sessionManager, this.dataService);
        this.emotionalSystem = new EmotionalSystem(this.dataService);
        this.debugTools = new DebugTools(this.apiUrlBase, this.dataService, this.sessionManager);
        
        // Initialize DOM references
        this.dom = {
            chatMessages: document.getElementById('chatMessages'),
            chatInput: document.getElementById('chatInput'),
            sendButton: document.getElementById('sendButton'),
            sessionInfo: document.getElementById('sessionInfo'),
            newSessionBtn: document.getElementById('newSessionBtn'),
            sessionsList: document.getElementById('sessionsList'),
            contextUsage: document.getElementById('contextUsage'),
        };
        
        // Clear any stale session references
        this.clearStaleReferences();
        
        // Defer initialization until the DOM is fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.init();
            });
        } else {
            // DOM is already loaded
            this.init();
        }
    }

    // ---------------------------------------------------------------------------
    // INITIALIZATION
    // ---------------------------------------------------------------------------

    clearStaleReferences() {
        // Note: We no longer clear session storage on init to preserve sessions across page refreshes
        // Only clear if explicitly requested by user or in error conditions
        console.log('Session persistence enabled - not clearing session storage');
    }

    async init() {
        console.log("Initializing Daemon Dashboard...");
        
        // Set initial connection status
        this.updateConnectionStatus(false);
        
        // Initialize modules
        await this.initializeModules();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Setup periodic updates
        this.setupIntervals();
        
        // Initial data fetch
        await this.performInitialDataFetch();
        
        console.log("Daemon Dashboard Initialized.");
    }

    async initializeModules() {
        // Link session manager and chat interface for bidirectional communication
        this.sessionManager.setChatInterface(this.chatInterface);
        
        // Initialize debug components
        this.debugTools.initializeDebugComponents();
        
        // Setup chat interface event handlers
        this.chatInterface.setupEventHandlers();
        
        console.log("All modules initialized with cross-module communication established.");
    }

    async performInitialDataFetch() {
        try {
            // Fetch core system status
            await this.fetchAndUpdateDaemonStatus();
            await this.fetchAndUpdateMemoryStats();
            await this.fetchAndUpdateSystemHealth();
            await this.fetchAndUpdateTokenUsage();
            
            // Fetch sessions and load the active one
            await this.sessionManager.fetchAndRenderSessions();
            
            // Initialize dashboard components
            await this.initializeDashboardComponents();
            
        } catch (error) {
            console.error('Error during initial data fetch:', error);
        }
    }

    async initializeDashboardComponents() {
        try {
            // Fetch initial data for dashboard components
            await this.dashboardComponents.fetchAndDisplayRecentMemories();
            await this.dashboardComponents.fetchAndDisplayRecentEmotions();
            await this.dashboardComponents.fetchAndDisplayRecentPersonalityChanges();
            
            // Fetch enhanced emotional system data
            await this.emotionalSystem.fetchAndDisplayDaemonThoughts();
            await this.emotionalSystem.fetchAndDisplayCurrentMood();
            await this.emotionalSystem.fetchAndDisplayUserAnalysis();
            
            // Fetch new emotional system data
            await this.emotionalSystem.fetchAndDisplayDetailedEmotionState();
            await this.emotionalSystem.fetchAndDisplayActiveSeeds();
            await this.emotionalSystem.fetchAndDisplayDistortionFrame();
            await this.emotionalSystem.fetchAndDisplayEmotionalMetrics();
            
        } catch (error) {
            console.error('Error initializing dashboard components:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // EVENT LISTENERS
    // ---------------------------------------------------------------------------

    setupEventListeners() {
        // New session button
        if (this.dom.newSessionBtn) {
            this.dom.newSessionBtn.addEventListener('click', () => {
                this.sessionManager.createNewSession();
            });
        }

        // Auto-resize textarea
        if (this.dom.chatInput) {
            this.dom.chatInput.addEventListener('input', () => {
                this.autoResizeTextarea(this.dom.chatInput);
            });
        }

        // Update send button state based on input
        this.chatInterface.updateSendButton();
        
        console.log("Core event listeners setup complete.");
    }

    autoResizeTextarea(textarea) {
        if (!textarea) return;
        
        // Reset height to auto to get the correct scrollHeight
        textarea.style.height = 'auto';
        
        // Set the height to the scrollHeight, with min and max constraints
        const minHeight = 40; // Minimum height in pixels
        const maxHeight = 120; // Maximum height in pixels
        const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
        
        textarea.style.height = newHeight + 'px';
        
        // Show/hide scrollbar based on content
        if (textarea.scrollHeight > maxHeight) {
            textarea.style.overflowY = 'auto';
        } else {
            textarea.style.overflowY = 'hidden';
        }
    }

    // ---------------------------------------------------------------------------
    // PERIODIC UPDATES
    // ---------------------------------------------------------------------------

    setupIntervals() {
        // Core system updates every 10 seconds
        this.updateIntervals.coreSystem = setInterval(async () => {
            await this.fetchAndUpdateDaemonStatus();
            await this.fetchAndUpdateTokenUsage();
            await this.fetchAndUpdateSystemHealth();
        }, 10000);

        // Dashboard components updates every 15 seconds
        this.updateIntervals.dashboardComponents = setInterval(async () => {
            await this.dashboardComponents.fetchAndDisplayRecentMemories();
            await this.dashboardComponents.fetchAndDisplayRecentEmotions();
            await this.dashboardComponents.fetchAndDisplayRecentPersonalityChanges();
        }, 15000);

        // Emotional system updates every 20 seconds
        this.updateIntervals.emotionalSystem = setInterval(async () => {
            await this.emotionalSystem.fetchAndDisplayCurrentMood();
            await this.emotionalSystem.fetchAndDisplayUserAnalysis();
            await this.emotionalSystem.fetchAndDisplayDaemonThoughts();
        }, 20000);

        // Advanced emotional system updates every 30 seconds
        this.updateIntervals.advancedEmotional = setInterval(async () => {
            await this.emotionalSystem.fetchAndDisplayDetailedEmotionState();
            await this.emotionalSystem.fetchAndDisplayActiveSeeds();
            await this.emotionalSystem.fetchAndDisplayDistortionFrame();
            await this.emotionalSystem.fetchAndDisplayEmotionalMetrics();
        }, 30000);

        // Memory stats update every 30 seconds
        this.updateIntervals.memoryStats = setInterval(async () => {
            await this.fetchAndUpdateMemoryStats();
        }, 30000);

        console.log("Update intervals established.");
    }

    cleanup() {
        // Clear all intervals
        Object.values(this.updateIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        this.updateIntervals = {};
        
        // Cleanup debug tools
        this.debugTools.cleanup();
        
        console.log("Dashboard cleanup completed.");
    }

    // ---------------------------------------------------------------------------
    // DATA FETCHING AND UI UPDATES
    // ---------------------------------------------------------------------------

    async fetchAndUpdateDaemonStatus() {
        try {
            const data = await this.dataService.fetchDaemonStatus();
            this.updateDaemonStatusUI(data);
            this.updateConnectionStatus(true);
        } catch (error) {
            console.error('Error fetching daemon status:', error);
            this.updateConnectionStatus(false);
        }
    }

    async fetchAndUpdateMemoryStats() {
        try {
            const data = await this.dataService.fetchMemoryStats();
            // Handle ChromaDB unavailable gracefully
            if (data && data.error && data.error.includes('ChromaDB not available')) {
                // Only log this once, not on every poll
                if (!this.chromaDbWarningShown) {
                    console.info('ℹ️ ChromaDB service unavailable - using fallback memory stats');
                    this.chromaDbWarningShown = true;
                }
                // Update UI with limited memory stats indicator
                this.updateMemoryStatsUI(data, true); // true = limited mode
            } else {
                console.log('✅ Memory stats updated:', data);
                this.chromaDbWarningShown = false;
                this.updateMemoryStatsUI(data, false);
            }
        } catch (error) {
            console.error('❌ Error fetching memory stats:', error);
        }
    }

    updateMemoryStatsUI(data, isLimited = false) {
        // Update memory stats display with limited mode indication
        const memoryElement = document.getElementById('memoryStats');
        if (memoryElement && isLimited) {
            const limitedIndicator = memoryElement.querySelector('.limited-indicator') || 
                document.createElement('div');
            limitedIndicator.className = 'limited-indicator';
            limitedIndicator.textContent = 'Limited (ChromaDB unavailable)';
            limitedIndicator.style.cssText = 'font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;';
            
            if (!memoryElement.querySelector('.limited-indicator')) {
                memoryElement.appendChild(limitedIndicator);
            }
        } else if (memoryElement) {
            const limitedIndicator = memoryElement.querySelector('.limited-indicator');
            if (limitedIndicator) {
                limitedIndicator.remove();
            }
        }
    }

    async fetchAndUpdateSystemHealth() {
        try {
            const data = await this.dataService.fetchHealthStatus();
            this.updateSystemHealthUI(data);
        } catch (error) {
            console.error('Error fetching system health:', error);
        }
    }

    async fetchAndUpdateTokenUsage() {
        try {
            const data = await this.dataService.fetchTokenUsage();
            this.dashboardComponents.updateTokenUsageDisplay(data, this.dom.contextUsage);
        } catch (error) {
            console.error('Error fetching token usage:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // UI UPDATE METHODS
    // ---------------------------------------------------------------------------

    updateDaemonStatusUI(data) {
        this.dashboardComponents.updateDaemonStatusUI(data, (statusData) => {
            // Custom update callback for daemon status
            this.updateStatusIndicators(statusData);
        });
    }

    updateStatusIndicators(data) {
        // Helper method to update generic status indicators
        const updateText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value !== null && value !== undefined ? value.toString() : 'N/A';
        };

        // Update recursion metrics if present
        if (data.recursion) {
            updateText('recursionPressure', (data.recursion.recursion_pressure || 0).toFixed(2));
            updateText('bufferCount', data.recursion.buffer_size || 0);
        }

        // Update shadow metrics if present  
        if (data.shadow) {
            updateText('shadowCharge', (data.shadow.integration_charge || 0).toFixed(2));
            updateText('shadowElements', data.shadow.active_elements || 0);
        }

        // Update personality metrics if present
        if (data.personality) {
            updateText('personalityEvolution', (data.personality.evolution_rate || 0).toFixed(2));
        }

        // Update advanced emotional metrics for dashboard coherence
        if (data.emotional_system) {
            const emotionData = data.emotional_system;
            
            // Update recursion pressure label with emotional context
            const pressureLabel = document.querySelector('#recursionPressure')?.parentElement.querySelector('.metric-label');
            if (pressureLabel && emotionData.processing_load) {
                pressureLabel.textContent = `Processing Load (${(emotionData.processing_load * 100).toFixed(0)}%):`;
            }
            
            // Update buffer count with emotional context
            const bufferLabel = document.querySelector('#bufferCount')?.parentElement.querySelector('.metric-label');
            if (bufferLabel && emotionData.active_seeds_count) {
                bufferLabel.textContent = `Active Seeds (${emotionData.active_seeds_count}):`;
            }
            
            // Update shadow charge with regulation efficiency
            const regulationEfficiency = emotionData.regulation_performance?.regulation_efficiency || 0.89;
            updateText('shadowCharge', regulationEfficiency.toFixed(2));
            
            const shadowLabel = document.querySelector('#shadowCharge')?.parentElement.querySelector('.metric-label');
            if (shadowLabel) {
                shadowLabel.textContent = `Regulation Efficiency:`;
            }
            
            // Update shadow elements with mood variety
            const moodVariety = emotionData.mood_diversity?.unique_moods || 8;
            updateText('shadowElements', moodVariety);
            
            const elementsLabel = document.querySelector('#shadowElements')?.parentElement.querySelector('.metric-label');
            if (elementsLabel) {
                elementsLabel.textContent = `Mood Variety:`;
            }
        }
    }

    updateSystemHealthUI(data) {
        const updateText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value !== null && value !== undefined ? value.toString() : 'N/A';
        };

        if (data.system_info) {
            const info = data.system_info;
            updateText('memoryUsage', info.memory_usage || 'N/A');
            updateText('cpuUsage', info.cpu_usage || 'N/A');
            updateText('diskSpace', info.disk_space || 'N/A');
        }
    }

    updateConnectionStatus(isConnected) {
        this.dashboardComponents.updateConnectionStatus(isConnected);
    }

    // ---------------------------------------------------------------------------
    // PUBLIC API
    // ---------------------------------------------------------------------------

    // Expose key functionality for external access
    getSessionManager() {
        return this.sessionManager;
    }

    getChatInterface() {
        return this.chatInterface;
    }

    getDashboardComponents() {
        return this.dashboardComponents;
    }

    getEmotionalSystem() {
        return this.emotionalSystem;
    }

    getDebugTools() {
        return this.debugTools;
    }

    // Legacy compatibility methods
    displayChatHistory() {
        return this.chatInterface.displayChatHistory();
    }

    async sendMessage() {
        return await this.chatInterface.sendMessage();
    }

    async setActiveSession(sessionId) {
        return await this.sessionManager.setActiveSession(sessionId);
    }

    async createNewSession() {
        return await this.sessionManager.createNewSession();
    }

    async deleteSession(sessionId) {
        return await this.sessionManager.deleteSession(sessionId);
    }
}

// Initialize the dashboard when the script loads
let dashboardInstance;

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardCore;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.DashboardCore = DashboardCore;
}