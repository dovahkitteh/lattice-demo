// Lattice AI Web Interface
class LatticeApp {
    constructor() {
        this.latticeUrl = 'http://127.0.0.1:11434';
        this.isConnected = false;
        this.isLoading = false;
        this.currentEmotions = {
            user: [],
            ai: []
        };
        this.emotionCharts = {
            user: null,
            ai: null
        };
        
        // GoEmotions label mapping
        this.emotionLabels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ];

        this.currentSessionId = null;
        this.sessions = {}; // Store session data
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Lattice AI Interface...');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize emotion charts
        this.initEmotionCharts();
        
        // Check connection and load initial data
        await this.checkConnection();
        
        if (this.isConnected) {
            await this.loadActiveSessions();
            this.startPeriodicUpdates();
        }
        
        console.log('✅ Lattice AI Interface ready!');
    }

    setupEventListeners() {
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const newConversationBtn = document.getElementById('newConversationBtn');
        
        messageInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.sendMessage();
            }
        });
        
        sendBtn.addEventListener('click', () => this.sendMessage());

        newConversationBtn.addEventListener('click', () => this.startNewConversation());
        
        messageInput.addEventListener('input', () => {
            sendBtn.disabled = !messageInput.value.trim();
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        });
    }

    async checkConnection() {
        try {
            console.log('🔍 Checking Lattice connection...');
            const response = await fetch(`${this.latticeUrl}/health`);
            
            if (response.ok) {
                this.isConnected = true;
                this.updateConnectionStatus(true, 'Connected');
                console.log('✅ Connected to Lattice service');
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Connection failed:', error);
            this.isConnected = false;
            this.updateConnectionStatus(false, 'Disconnected');
        }
    }

    updateConnectionStatus(connected, text) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = `status-dot ${connected ? 'connected' : ''}`;
        statusText.textContent = text;
        
        // Enable/disable send button based on connection
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');
        if (!connected) {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = !messageInput.value.trim();
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message || this.isLoading) return;

        this.isLoading = true;
        
        // Clear input and disable send button
        messageInput.value = '';
        messageInput.style.height = 'auto';
        document.getElementById('sendBtn').disabled = true;

        // Add user message to chat
        this.addMessageToChat(message, 'user');

        // Create placeholder for AI response
        const aiMessageDiv = this.createStreamingMessage();

        try {
            console.log('📤 Sending message to Lattice...');
            
            // Get current session ID
            const sessionID = this.getCurrentSessionId();
            if (!sessionID) {
                console.error("No active session ID found!");
                this.finalizeStreamingMessage(aiMessageDiv, `Error: No active session. Please start a new conversation.`, true);
                return;
            }

            const response = await fetch(`${this.latticeUrl}/v1/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionID,
                    model: 'mistral',
                    messages: [{ role: 'user', content: message }],
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiResponse = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;
                            
                            try {
                                const parsed = JSON.parse(data);
                                const content = parsed.choices[0]?.delta?.content || '';
                                if (content) {
                                    aiResponse += content;
                                    this.updateStreamingMessage(aiMessageDiv, aiResponse);
                                }
                            } catch (e) {
                                // Skip malformed JSON
                            }
                        }
                    }
                }
            } catch (streamError) {
                console.error('Streaming error:', streamError);
            }

            // Finalize the message
            this.finalizeStreamingMessage(aiMessageDiv, aiResponse || 'No response generated.');
            
            console.log('✅ Message sent and response received');
            
            // Update emotions and memories after successful response
            setTimeout(() => {
                if (this.isConnected && !this.isLoading) {
                    this.loadMemoryStats();
                    this.loadRecentMemories();
                }
            }, 3000);
            
        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.finalizeStreamingMessage(aiMessageDiv, `Error: ${error.message}`, true);
        } finally {
            this.isLoading = false;
        }
    }

    addMessageToChat(content, type) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (type === 'user') {
            contentDiv.textContent = content;
        } else if (type === 'bot') {
            contentDiv.innerHTML = `<strong>Lattice AI:</strong> ${content}`;
        } else if (type === 'error') {
            contentDiv.innerHTML = `<strong>Error:</strong> ${content}`;
            messageDiv.className = 'message bot-message error';
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    createStreamingMessage() {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<strong>Lattice AI:</strong> <span class="typing-indicator">●</span>';
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }

    updateStreamingMessage(messageDiv, content) {
        const contentDiv = messageDiv.querySelector('.message-content');
        contentDiv.innerHTML = `<strong>Lattice AI:</strong> ${content}`;
        
        // Scroll to bottom
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    finalizeStreamingMessage(messageDiv, content, isError = false) {
        const contentDiv = messageDiv.querySelector('.message-content');
        
        if (isError) {
            contentDiv.innerHTML = `<strong>Error:</strong> ${content}`;
            messageDiv.classList.add('error');
        } else {
            // Use a safer method to set content
            const strong = document.createElement('strong');
            strong.textContent = 'Lattice AI: ';
            contentDiv.innerHTML = ''; // Clear previous content
            contentDiv.appendChild(strong);
            contentDiv.append(document.createTextNode(content));
        }
    }

    async loadMemoryStats() {
        try {
            console.log('📊 Loading memory statistics...');
            const response = await fetch(`${this.latticeUrl}/v1/memories/stats`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update memory stats
            document.getElementById('totalMemories').textContent = data.total_memories || 0;
            document.getElementById('chromaStatus').textContent = 
                data.database_components?.chroma_db ? '✅' : '❌';
            document.getElementById('neo4jStatus').textContent = 
                data.database_components?.neo4j_conn ? '✅' : '❌';
            
            // Update emotions from latest memory
            if (data.sample_memories?.metadatas?.length > 0) {
                this.updateEmotionsFromMemory(data.sample_memories.metadatas[0]);
            }
            
            console.log('✅ Memory stats updated');
            
        } catch (error) {
            console.error('❌ Error loading memory stats:', error);
        }
    }

    async loadRecentMemories() {
        try {
            const response = await fetch(`${this.latticeUrl}/v1/memories/recent`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            const memoriesList = document.getElementById('recentMemoriesList');
            memoriesList.innerHTML = ''; // Clear only the list

            if (data.memories && data.memories.length > 0) {
                data.memories.forEach(memory => {
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <p><strong>Synopsis:</strong> ${memory.synopsis || 'N/A'}</p>
                        <p class="timestamp">${new Date(memory.timestamp).toLocaleString()}</p>
                    `;
                    memoriesList.appendChild(li);
                });
            } else {
                memoriesList.innerHTML = '<li>No recent memories found.</li>';
            }
        } catch (error) {
            console.error('❌ Error loading recent memories:', error);
            const memoriesList = document.getElementById('recentMemoriesList');
            memoriesList.innerHTML = '<li>Error loading memories.</li>';
        }
    }

    updateEmotionsFromMemory(metadata) {
        if (!metadata) return;
        try {
            // Parse emotion vectors
            const userAffect = metadata.user_affect ? JSON.parse(metadata.user_affect) : null;
            const selfAffect = metadata.self_affect ? JSON.parse(metadata.self_affect) : null;
            
            if (userAffect) {
                this.updateEmotionDisplay('user', userAffect);
            }
            
            if (selfAffect) {
                this.updateEmotionDisplay('ai', selfAffect);
            }
            
            console.log('✅ Emotions updated from memory');
            
        } catch (error) {
            console.error('❌ Error updating emotions:', error);
        }
    }

    updateEmotionDisplay(type, emotionVector) {
        // Find top emotions
        const topEmotions = emotionVector
            .map((score, index) => ({ emotion: this.emotionLabels[index], score }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
        
        // Update emotion list
        const listElement = document.getElementById(`${type}EmotionList`);
        listElement.innerHTML = '';
        
        topEmotions.forEach(({ emotion, score }) => {
            if (score > 0.01) { // Only show emotions with score > 1%
                const tag = document.createElement('span');
                tag.className = `emotion-tag ${score > 0.1 ? 'strong' : ''}`;
                tag.textContent = `${emotion} ${(score * 100).toFixed(1)}%`;
                listElement.appendChild(tag);
            }
        });
        
        // Update chart
        this.updateEmotionChart(type, topEmotions);
    }

    initEmotionCharts() {
        // Disable Chart.js initialization to prevent interface issues
        console.log('⚠️ Chart.js disabled to prevent interface issues');
        
        // Hide chart containers
        try {
            const userChart = document.getElementById('userEmotionChart');
            const aiChart = document.getElementById('aiEmotionChart');
            if (userChart) userChart.style.display = 'none';
            if (aiChart) aiChart.style.display = 'none';
        } catch (error) {
            console.error('Error hiding chart containers:', error);
        }
    }

    updateEmotionChart(type, topEmotions) {
        const chart = this.emotionCharts[type];
        if (!chart) return;

        const validEmotions = topEmotions.filter(e => e.score > 0.01);
        
        chart.data.labels = validEmotions.map(e => e.emotion);
        chart.data.datasets[0].data = validEmotions.map(e => e.score);
        chart.update();
    }

    updateReflections(metadata) {
        const userReflection = document.getElementById('latestUserReflection');
        const aiReflection = document.getElementById('latestAiReflection');
        
        userReflection.textContent = metadata.user_reflection || 'No recent user reflection';
        aiReflection.textContent = metadata.self_reflection || 'No recent AI reflection';
    }

    startPeriodicUpdates() {
        // Disable automatic periodic updates to prevent infinite growth issues
        // Updates will only happen after successful message sends
        console.log('⚠️ Periodic updates disabled to prevent interface issues');
    }

    showLoadingOverlay(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }

    async loadActiveSessions() {
        try {
            const response = await fetch(`${this.latticeUrl}/v1/conversations/active`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.sessions = await response.json();
            this.renderSessionList();
            
            // If no active session, or current one is invalid, select the first or create new
            const sessionIds = Object.keys(this.sessions);
            if (!this.currentSessionId || !this.sessions[this.currentSessionId]) {
                if (sessionIds.length > 0) {
                    await this.selectSession(sessionIds[0]);
                } else {
                    await this.startNewConversation();
                }
            } else {
                 await this.selectSession(this.currentSessionId);
            }
        } catch (error) {
            console.error('❌ Error loading active sessions:', error);
            // Attempt to start a new conversation as a fallback
            await this.startNewConversation();
        }
    }
    
    renderSessionList() {
        const sessionList = document.getElementById('sessionList');
        sessionList.innerHTML = '';
        const sortedSessions = Object.values(this.sessions).sort((a, b) => 
            new Date(b.last_activity) - new Date(a.last_activity)
        );

        for (const session of sortedSessions) {
            const li = document.createElement('li');
            li.dataset.sessionId = session.id;
            li.className = session.id === this.currentSessionId ? 'active' : '';
            li.textContent = session.title;
            li.addEventListener('click', () => this.selectSession(session.id));
            sessionList.appendChild(li);
        }
    }

    async selectSession(sessionId) {
        if (!this.sessions[sessionId] || this.isLoading) return;

        this.currentSessionId = sessionId;
        this.renderSessionList(); // Re-render to show active state
        
        this.showLoadingOverlay(true);
        try {
            const response = await fetch(`${this.latticeUrl}/v1/conversations/sessions/${sessionId}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const sessionData = await response.json();
            
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = ''; // Clear previous messages
            
            if (sessionData.history) {
                sessionData.history.forEach(turn => {
                    this.addMessageToChat(turn.user, 'user');
                    this.addMessageToChat(turn.ai, 'bot');
                });
            }
            
            await this.loadMemoryStats();
            await this.loadRecentMemories();

        } catch (error) {
            console.error(`❌ Error selecting session ${sessionId}:`, error);
            this.addMessageToChat(`Error loading session.`, 'error');
        } finally {
            this.showLoadingOverlay(false);
        }
    }

    async startNewConversation() {
        if (this.isLoading) return;
        this.showLoadingOverlay(true);

        try {
            const response = await fetch(`${this.latticeUrl}/v1/conversations/sessions/new`, { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const newSession = await response.json();
            
            // Add to our sessions list and select it
            this.sessions[newSession.id] = newSession;
            await this.selectSession(newSession.id);

        } catch (error) {
            console.error('❌ Error starting new conversation:', error);
            this.addMessageToChat('Could not start a new conversation. Please check the server.', 'error');
        } finally {
            this.showLoadingOverlay(false);
        }
    }

    getCurrentSessionId() {
        return this.currentSessionId;
    }

    // --- Periodic Updates ---
    startPeriodicUpdates() {
        this.updateInterval = setInterval(() => {
            if (this.isConnected && !this.isLoading) {
                this.loadMemoryStats();
                this.loadRecentMemories();
            }
        }, 5000); // Update every 5 seconds
    }
}

// Global functions for HTML event handlers
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    if (window.latticeApp) {
        window.latticeApp.sendMessage();
    }
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="message bot-message">
                <div class="message-content">
                    <strong>Lattice AI:</strong> Chat cleared! I'm ready for a new conversation.
                </div>
            </div>
        </div>
    `;
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.latticeApp = new LatticeApp();
}); 