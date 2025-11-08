/**
 * Chat Interface Module
 * Handles real-time messaging, streaming, and file uploads for the Daemon Dashboard
 */

class ChatInterface {
    constructor(apiUrlBase, sessionManager, dataService) {
        this.apiUrlBase = apiUrlBase;
        this.sessionManager = sessionManager;
        this.dataService = dataService;
        this.isLoading = false;
        this.conversationHistory = [];
        this.lastActivity = Date.now();
        this.utils = DashboardUtils;
    }

    // ---------------------------------------------------------------------------
    // CHAT DISPLAY AND HISTORY
    // ---------------------------------------------------------------------------

    displayChatHistory() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.error('chatMessages element not found');
            return;
        }
        
        chatMessages.innerHTML = '';
        
        // Ensure conversationHistory is initialized and is an array
        if (!this.conversationHistory || !Array.isArray(this.conversationHistory)) {
            console.warn('conversationHistory not properly initialized, using empty array');
            this.conversationHistory = [];
        }
        
        // Filter out system messages and only show user/assistant messages
        const displayableMessages = this.conversationHistory.filter(msg => 
            msg && msg.role && (msg.role === 'user' || msg.role === 'assistant')
        );
        
        displayableMessages.forEach(message => {
            this.addMessageToChat(message.content, message.role, false); // false = don't auto-scroll
        });
        
        // If no messages, show empty state
        if (displayableMessages.length === 0) {
            chatMessages.innerHTML = '<div class="empty-state">◆ Begin conversation to witness recursive evolution ◆</div>';
        } else {
            // Auto-scroll to bottom after displaying all messages
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    addMessageToChat(content, role, autoScroll = true) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.error('chatMessages element not found');
            return null;
        }

        // Remove empty state if it exists
        const emptyState = chatMessages.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message-container ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Handle different content types and formatting
        if (typeof content === 'string') {
            let formattedContent = content;
            
            // Handle code blocks
            if (content.includes('```')) {
                formattedContent = content.replace(/```(\w+)?\n([\s\S]*?)```/g, 
                    '<pre><code class="language-$1">$2</code></pre>');
            }
            
            // Handle inline code
            formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Handle line breaks
            formattedContent = formattedContent.replace(/\n/g, '<br>');
            
            // Handle bold text
            formattedContent = formattedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // Handle italic text
            formattedContent = formattedContent.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            
            messageContent.innerHTML = formattedContent;
        } else {
            messageContent.textContent = 'Invalid message content';
        }

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        if (autoScroll) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        return messageDiv;
    }

    showErrorInChat(message) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const errorElement = document.createElement('div');
        errorElement.className = 'message-container';
        errorElement.innerHTML = `<div class="message error-message"><strong>Error:</strong> ${message}</div>`;
        chatMessages.appendChild(errorElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '<div class="empty-state">◆ Begin conversation to witness recursive evolution ◆</div>';
        }
        this.conversationHistory = [];
    }

    // ---------------------------------------------------------------------------
    // MESSAGE SENDING AND STREAMING
    // ---------------------------------------------------------------------------

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput) {
            console.error('chatInput element not found');
            return;
        }
        
        const message = chatInput.value.trim();
        
        if (!message || this.isLoading) return;

        // Get active session ID
        const activeSessionId = this.sessionManager.getActiveSessionId();
        if (!activeSessionId) {
            this.showErrorInChat("Cannot send message: No active session. Please create or select a session.");
            return;
        }

        // Update activity timestamp
        this.lastActivity = Date.now();
        
        this.isLoading = true;
        this.updateSendButton();
        let sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.textContent = 'Sending...';
        }

        // Add user message to chat immediately
        this.addMessageToChat(message, 'user');
        
        // Add user message to conversation history immediately
        this.conversationHistory.push({
            role: 'user',
            content: message,
            timestamp: Date.now()
        });
        
        // Clear input
        chatInput.value = '';
        
        // Refresh the session list to update timestamp
        await this.sessionManager.refreshSessionsList();

        try {
            // Build conversation messages for API
            const conversationMessages = [];
            
            // Add recent conversation history (limit to prevent token overflow)
            const recentHistory = this.conversationHistory.slice(-10); // Last 10 messages
            recentHistory.forEach(msg => {
                if (msg.role === 'user' || msg.role === 'assistant') {
                    conversationMessages.push({
                        role: msg.role,
                        content: msg.content
                    });
                }
            });

            // Make API call
            const response = await fetch(`${this.apiUrlBase}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: conversationMessages,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`Chat API error: ${response.status}`);
            }

            if (response.body) {
                // Handle streaming response
                const streamingMessageDiv = this.createStreamingMessage();
                let aiMessage = '';
                
                try {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') continue;
                                
                                try {
                                    const parsed = JSON.parse(data);
                                    const content = parsed.choices[0]?.delta?.content || '';
                                    if (content) {
                                        aiMessage += content;
                                        this.updateStreamingMessage(streamingMessageDiv, aiMessage);
                                    }
                                } catch (parseError) {
                                    console.warn('Error parsing streaming chunk:', parseError);
                                }
                            }
                        }
                    }
                    
                    // Finalize the streaming message
                    this.finalizeStreamingMessage(streamingMessageDiv, aiMessage);
                    
                    // Add to conversation history
                    this.conversationHistory.push({
                        role: 'assistant',
                        content: aiMessage,
                        timestamp: Date.now()
                    });
                    
                } catch (streamError) {
                    console.error('Streaming error:', streamError);
                    this.finalizeStreamingMessage(streamingMessageDiv, aiMessage || 'Stream interrupted');
                }
            }

            // Refresh session info and token usage
            try {
                const activeSessionResponse = await fetch(`${this.apiUrlBase}/conversations/active`);
                if (activeSessionResponse.ok) {
                    const activeSessionData = await activeSessionResponse.json();
                    this.sessionManager.updateSessionInfo(activeSessionData);
                }
                
                const tokenUsageResponse = await fetch(`${this.apiUrlBase}/dashboard/token-usage`);
                if (tokenUsageResponse.ok) {
                    const tokenData = await tokenUsageResponse.json();
                    this.updateTokenDisplay(tokenData);
                }
            } catch (updateError) {
                console.warn('Error updating session info:', updateError);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.showErrorInChat(`Failed to send message: ${error.message}`);
            
            // Remove the user message from history if sending failed
            const currentSession = { session_id: activeSessionId, title: 'Active Session' };
            this.sessionManager.updateSessionInfo(currentSession);
        }

        this.isLoading = false;
        this.updateSendButton();
        // Reset send button text
        const btnElement = document.getElementById('sendButton');
        if (btnElement) {
            btnElement.textContent = 'Send';
        }
    }

    updateSendButton() {
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        
        if (chatInput && sendButton) {
            const hasText = chatInput.value.trim().length > 0;
            sendButton.disabled = this.isLoading || !hasText;
        }
    }

    updateTokenDisplay(tokenData) {
        const contextUsage = document.getElementById('contextUsage');
        if (contextUsage && tokenData) {
            const tokens = tokenData.context_tokens || 0;
            const maxTokens = tokenData.max_context || 8192;
            const percentage = tokenData.usage_percentage || 0;
            
            contextUsage.textContent = `Context: ${tokens}/${maxTokens} tokens (${percentage}%)`;
            
            // Add visual feedback for high usage
            if (percentage > 80) {
                contextUsage.style.color = 'var(--error-color)';
            } else if (percentage > 60) {
                contextUsage.style.color = 'var(--warning-color)';
            } else {
                contextUsage.style.color = 'var(--text-secondary)';
            }
        }
    }

    // ---------------------------------------------------------------------------
    // STREAMING MESSAGE HANDLING
    // ---------------------------------------------------------------------------

    createStreamingMessage() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return null;

        // Remove empty state if it exists
        const emptyState = chatMessages.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-container assistant streaming';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = '<span class="typing-indicator">●●●</span>';

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }

    updateStreamingMessage(messageDiv, content) {
        if (!messageDiv) return;
        
        const messageContent = messageDiv.querySelector('.message-content');
        if (messageContent) {
            // Apply same formatting as addMessageToChat
            let formattedContent = content;
            
            // Handle code blocks
            if (content.includes('```')) {
                formattedContent = content.replace(/```(\w+)?\n([\s\S]*?)```/g, 
                    '<pre><code class="language-$1">$2</code></pre>');
            }
            
            // Handle inline code
            formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Handle line breaks
            formattedContent = formattedContent.replace(/\n/g, '<br>');
            
            // Handle bold text
            formattedContent = formattedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // Handle italic text
            formattedContent = formattedContent.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            
            messageContent.innerHTML = formattedContent + '<span class="typing-cursor">|</span>';
        }

        // Auto-scroll to bottom
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    finalizeStreamingMessage(messageDiv, finalContent) {
        if (!messageDiv) return;
        
        const messageContent = messageDiv.querySelector('.message-content');
        if (messageContent && finalContent) {
            // Apply final formatting
            let formattedContent = finalContent;
            
            // Handle code blocks
            if (finalContent.includes('```')) {
                formattedContent = finalContent.replace(/```(\w+)?\n([\s\S]*?)```/g, 
                    '<pre><code class="language-$1">$2</code></pre>');
            }
            
            // Handle inline code
            formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Handle line breaks
            formattedContent = formattedContent.replace(/\n/g, '<br>');
            
            // Handle bold text
            formattedContent = formattedContent.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // Handle italic text
            formattedContent = formattedContent.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            
            messageContent.innerHTML = formattedContent;
        }

        // Remove streaming class
        messageDiv.classList.remove('streaming');
    }

    // ---------------------------------------------------------------------------
    // FILE UPLOAD FUNCTIONALITY
    // ---------------------------------------------------------------------------

    async handleSeedFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const uploadBtn = document.getElementById('uploadSeedBtn');
        const originalText = uploadBtn.textContent;

        try {
            // Show loading state
            uploadBtn.textContent = 'Processing...';
            uploadBtn.disabled = true;

            const text = await this.readFileAsText(file);
            
            // Validate file content
            if (!text.trim()) {
                throw new Error('File appears to be empty');
            }

            if (text.length > 50000) { // 50KB limit
                throw new Error('File too large (max 50KB)');
            }

            // Send to API
            const response = await fetch(`${this.apiUrlBase}/memories/upload-seed`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    seed_content: text,
                    filename: file.name
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
            // Show success feedback
            uploadBtn.textContent = '✓ Uploaded';
            uploadBtn.style.backgroundColor = 'var(--success-color)';
            
            // Add success message to chat
            this.showSuccessInChat(`Emotional seed uploaded successfully: ${file.name}`);
            
            // Reset after 3 seconds
            setTimeout(() => {
                uploadBtn.textContent = originalText;
                uploadBtn.style.backgroundColor = '';
                uploadBtn.disabled = false;
            }, 3000);

        } catch (error) {
            console.error('File upload error:', error);
            uploadBtn.textContent = '✗ Failed';
            uploadBtn.style.backgroundColor = 'var(--error-color)';
            
            // Show error in chat
            this.showErrorInChat(`Upload failed: ${error.message}`);
            
            // Reset after 3 seconds
            setTimeout(() => {
                uploadBtn.textContent = originalText;
                uploadBtn.style.backgroundColor = '';
                uploadBtn.disabled = false;
            }, 3000);
        }

        // Clear file input
        event.target.value = '';
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    showSuccessInChat(message) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const successElement = document.createElement('div');
        successElement.className = 'message-container';
        successElement.innerHTML = `<div class="message success-message"><strong>Success:</strong> ${message}</div>`;
        chatMessages.appendChild(successElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ---------------------------------------------------------------------------
    // EVENT HANDLERS SETUP
    // ---------------------------------------------------------------------------

    setupEventHandlers() {
        // Chat input event handlers
        const chatInput = document.getElementById('chatInput');
        const sendButtonEl = document.getElementById('sendButton');
        
        if (chatInput) {
            chatInput.addEventListener('input', () => this.updateSendButton());
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        if (sendButtonEl) {
            sendButtonEl.addEventListener('click', () => this.sendMessage());
        }

        // File upload event handlers
        const seedFileInput = document.getElementById('seedFileInput');
        const uploadSeedBtn = document.getElementById('uploadSeedBtn');

        if (uploadSeedBtn && seedFileInput) {
            uploadSeedBtn.addEventListener('click', () => seedFileInput.click());
        }

        if (seedFileInput) {
            seedFileInput.addEventListener('change', (e) => this.handleSeedFileUpload(e));
        }
    }

    // ---------------------------------------------------------------------------
    // UTILITY METHODS
    // ---------------------------------------------------------------------------

    getConversationHistory() {
        return this.conversationHistory;
    }

    setConversationHistory(history) {
        this.conversationHistory = Array.isArray(history) ? history : [];
        console.log(`📝 Chat interface: Set conversation history with ${this.conversationHistory.length} messages`);
        this.displayChatHistory();
    }

    // Load and display session messages
    async loadSessionMessages(sessionId) {
        console.log(`📂 Loading session messages for: ${sessionId}`);
        
        try {
            const response = await fetch(`${this.apiUrlBase}/conversations/sessions/${sessionId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch session details: ${response.status}`);
            }
            
            const sessionData = await response.json();
            console.log(`📋 Session data loaded: ${sessionData.messages?.length || 0} messages`);
            
            if (sessionData.messages && Array.isArray(sessionData.messages)) {
                this.setConversationHistory(sessionData.messages);
                console.log(`✅ Successfully loaded ${sessionData.messages.length} messages for session ${sessionId}`);
            } else {
                console.warn('⚠️ No messages found in session data');
                this.setConversationHistory([]);
            }
        } catch (error) {
            console.error('❌ Error loading session messages:', error);
            this.setConversationHistory([]);
        }
    }

    getLastActivity() {
        return this.lastActivity;
    }

    updateLastActivity() {
        this.lastActivity = Date.now();
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatInterface;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.ChatInterface = ChatInterface;
}