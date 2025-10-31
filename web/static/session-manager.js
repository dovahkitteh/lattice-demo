/**
 * Session Manager Module
 * Handles conversation session management for the Daemon Dashboard
 */

class SessionManager {
    constructor(sessionsApiUrl, dataService) {
        this.sessionsApiUrl = sessionsApiUrl;
        this.dataService = dataService;
        this.activeSessionId = this.loadActiveSessionFromStorage();
        this.utils = DashboardUtils;
        this.chatInterface = null; // Will be set by dashboard core
    }

    // Set chat interface reference for bidirectional communication
    setChatInterface(chatInterface) {
        this.chatInterface = chatInterface;
        console.log('🔗 Session manager: Chat interface linked');
    }

    // Load active session from localStorage
    loadActiveSessionFromStorage() {
        try {
            return localStorage.getItem('lattice_active_session_id') || null;
        } catch (error) {
            console.warn('Could not load session from storage:', error);
            return null;
        }
    }

    // Save active session to localStorage
    saveActiveSessionToStorage(sessionId) {
        try {
            if (sessionId) {
                localStorage.setItem('lattice_active_session_id', sessionId);
            } else {
                localStorage.removeItem('lattice_active_session_id');
            }
        } catch (error) {
            console.warn('Could not save session to storage:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // SESSION FETCHING AND RENDERING
    // ---------------------------------------------------------------------------

    async fetchAndRenderSessions() {
        try {
            const response = await fetch(this.sessionsApiUrl);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            
            // Check if the API returned an error
            if (data.error) {
                console.error('API error fetching sessions:', data.error);
                this.displaySessionsError(data.error);
                return;
            }
            
            // Ensure sessions is an array
            const sessions = Array.isArray(data.sessions) ? data.sessions : [];
            console.log(`🔍 fetchAndRenderSessions: API returned ${sessions.length} sessions`);
            this.renderSessionsList(sessions);
            
            // Only set active session if we don't have one already set
            if (!this.activeSessionId) {
                await this.initializeActiveSession(sessions);
            }

        } catch (error) {
            console.error('Error fetching sessions:', error);
            this.displaySessionsError('Could not load sessions');
        }
    }

    async initializeActiveSession(sessions) {
        console.log(`🎯 Initializing active session. Chat interface linked: ${!!this.chatInterface}`);
        console.log(`💾 Stored session ID: ${this.activeSessionId}`);
        console.log(`📊 Available sessions: ${sessions.length}`);
        
        // First, try to restore from localStorage if we have a stored session
        if (this.activeSessionId) {
            const storedSession = sessions.find(s => s.session_id === this.activeSessionId);
            if (storedSession) {
                console.log(`🔄 Restoring active session from storage: ${this.activeSessionId}`);
                await this.setActiveSession(this.activeSessionId);
                return;
            } else {
                console.log('⚠️ Stored session not found on server, clearing storage');
                this.activeSessionId = null;
                this.saveActiveSessionToStorage(null);
            }
        }

        // Try to load active session from server, otherwise use the most recent
        const activeSession = sessions.find(s => s.is_active);
        if (activeSession) {
            console.log(`🎯 Using server active session: ${activeSession.session_id}`);
            await this.setActiveSession(activeSession.session_id);
        } else if (sessions.length > 0) {
            // Fallback to the most recent session
            console.log(`📅 Using most recent session: ${sessions[0].session_id}`);
            await this.setActiveSession(sessions[0].session_id);
        } else {
            // No sessions exist, create one
            console.log('➕ No sessions found, creating new session');
            await this.createNewSession();
        }
    }

    renderSessionsList(sessions) {
        const sessionsList = document.getElementById('sessionsList');
        if (!sessionsList) return;

        sessionsList.innerHTML = '';
        
        // Defensive programming: ensure sessions is an array
        if (!Array.isArray(sessions)) {
            console.warn('renderSessionsList received non-array data:', sessions);
            this.displaySessionsError('Invalid sessions data format');
            return;
        }
        
        if (sessions.length === 0) {
            sessionsList.innerHTML = '<div class="empty-state">No past conversations.</div>';
            return;
        }

        console.log(`🎯 Rendering ${sessions.length} sessions to dashboard`);
        
        let renderedCount = 0;
        sessions.forEach((session, index) => {
            const item = this.createSessionItem(session);
            sessionsList.appendChild(item);
            renderedCount++;
        });

        console.log(`✅ Successfully rendered ${renderedCount} sessions`);
    }

    createSessionItem(session) {
        const item = document.createElement('div');
        item.className = `session-item ${session.session_id === this.activeSessionId ? 'active' : ''}`;
        item.dataset.sessionId = session.session_id;

        const header = document.createElement('div');
        header.className = 'session-header';

        const content = document.createElement('div');
        content.className = 'session-content';

        const title = document.createElement('div');
        title.className = 'session-title';
        // Use stored rename if available, otherwise use server title
        const storedTitle = this.getSessionRename(session.session_id);
        title.textContent = storedTitle || session.title || 'Untitled Session';
        
        const timestamp = document.createElement('div');
        timestamp.className = 'session-timestamp';
        timestamp.textContent = `Last activity: ${new Date(session.last_updated).toLocaleString()}`;

        content.appendChild(title);
        content.appendChild(timestamp);

        const actions = document.createElement('div');
        actions.className = 'session-actions';
        
        const renameBtn = document.createElement('button');
        renameBtn.className = 'session-action-btn rename-btn';
        renameBtn.innerHTML = '✏️';
        renameBtn.title = 'Rename session';
        renameBtn.onclick = (e) => {
            e.stopPropagation();
            this.renameSession(session.session_id, title);
        };
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'session-action-btn delete-btn';
        deleteBtn.innerHTML = '🗑️';
        deleteBtn.title = 'Delete session';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            this.deleteSession(session.session_id);
        };

        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);
        
        header.appendChild(content);
        header.appendChild(actions);
        item.appendChild(header);

        item.onclick = () => {
            this.setActiveSession(session.session_id);
        };

        return item;
    }

    displaySessionsError(message) {
        const sessionsList = document.getElementById('sessionsList');
        if (sessionsList) {
            sessionsList.innerHTML = `<div class="error">Error: ${message}</div>`;
        }
    }

    // ---------------------------------------------------------------------------
    // SESSION CRUD OPERATIONS
    // ---------------------------------------------------------------------------

    async setActiveSession(sessionId) {
        try {
            console.log(`🔄 Setting active session: ${sessionId}`);
            
            // Make API call to set active session
            const response = await fetch(`${this.sessionsApiUrl}/${sessionId}/set_active`, { method: 'POST' });
            if (!response.ok) {
                throw new Error(`Failed to set active session: ${response.status}`);
            }

            // Update local state and persist to storage
            this.activeSessionId = sessionId;
            this.saveActiveSessionToStorage(sessionId);
            
            // Get session details for display and load messages
            const detailsResponse = await fetch(`${this.sessionsApiUrl}/${sessionId}`);
            if (detailsResponse.ok) {
                const sessionDetails = await detailsResponse.json();
                this.updateSessionInfo(sessionDetails);
                console.log(`✅ Successfully set active session: ${sessionId}`);
                
                // Load session messages in chat interface
                if (this.chatInterface) {
                    console.log(`📂 Loading messages for active session: ${sessionId}`);
                    // Small delay to ensure DOM is ready
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await this.chatInterface.loadSessionMessages(sessionId);
                } else {
                    console.warn('⚠️ Chat interface not linked - cannot load session messages');
                    // Try to load later if chat interface becomes available
                    setTimeout(async () => {
                        if (this.chatInterface) {
                            console.log('🔄 Retry: Loading messages for active session after delay');
                            await this.chatInterface.loadSessionMessages(sessionId);
                        }
                    }, 500);
                }
            }

            // Refresh the sessions list to update active status
            const sessionsResponse = await fetch(this.sessionsApiUrl);
            const data = await sessionsResponse.json();
            if (data && !data.error) {
                // Update UI to reflect active session
                this.updateActiveSessionUI(sessionId);
                
                // Ensure sessions is an array before processing
                const sessions = Array.isArray(data.sessions) ? data.sessions : [];
                this.renderSessionsList(sessions);
            }

            return true;
        } catch (error) {
            console.error('Error setting active session:', error);
            this.displaySessionError(`Failed to set active session: ${error.message}`);
            return false;
        }
    }

    async createNewSession() {
        try {
            console.log('🆕 Creating new session...');
            
            const response = await fetch(`${this.sessionsApiUrl}/new`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: 'New Conversation'
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to create new session: ${response.status}`);
            }
            
            const newSession = await response.json();
            console.log('✅ New session created:', newSession);
            
            // Update local state and persist to storage
            this.activeSessionId = newSession.session_id;
            this.saveActiveSessionToStorage(newSession.session_id);
            
            // Create session object for display
            const sessionForDisplay = {
                session_id: newSession.session_id,
                title: 'New Conversation',
                last_updated: new Date().toISOString(),
                is_active: true,
                turn_count: 0
            };
            
            // Refresh sessions list
            const sessionsResponse = await fetch(this.sessionsApiUrl);
            const data = await sessionsResponse.json();
            if (data && !data.error) {
                // Update session info
                this.updateSessionInfo(sessionForDisplay);
                
                // Ensure sessions is an array before processing
                const sessions = Array.isArray(data.sessions) ? data.sessions : [];
                this.renderSessionsList(sessions);
            }

            return newSession;
        } catch (error) {
            console.error('Error creating new session:', error);
            this.displaySessionError(`Failed to create new session: ${error.message}`);
            return null;
        }
    }

    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
            return;
        }

        try {
            console.log(`🗑️ Deleting session: ${sessionId}`);
            
            const response = await fetch(`${this.sessionsApiUrl}/${sessionId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`Failed to delete session: ${response.status}`);
            }

            console.log('✅ Session deleted successfully');

            // If we deleted the active session, clear it
            if (this.activeSessionId === sessionId) {
                this.activeSessionId = null;
                this.saveActiveSessionToStorage(null);
                this.clearSessionInfo();
            }

            // Refresh sessions list
            await this.fetchAndRenderSessions();

        } catch (error) {
            console.error('Error deleting session:', error);
            this.displaySessionError(`Failed to delete session: ${error.message}`);
        }
    }

    async renameSession(sessionId, titleElement) {
        const currentTitle = titleElement.textContent;
        const newTitle = prompt('Enter new session name:', currentTitle);
        
        if (!newTitle || newTitle.trim() === '' || newTitle === currentTitle) {
            return; // User cancelled or didn't change the title
        }

        try {
            console.log(`✏️ Renaming session ${sessionId} to: ${newTitle}`);
            
            // Store the rename in localStorage for persistence
            this.storeSessionRename(sessionId, newTitle.trim());
            
            // Update the UI immediately
            titleElement.textContent = newTitle.trim();
            
            // Optional: You could add an API call here when the backend supports it
            /*
            const response = await fetch(`${this.apiUrlBase}/conversations/sessions/${sessionId}/rename`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: newTitle.trim() })
            });

            if (!response.ok) {
                throw new Error(`Failed to rename session: ${response.status}`);
            }
            */

            console.log('✅ Session renamed successfully');
            
        } catch (error) {
            console.error('Error renaming session:', error);
            // Revert the title change on error
            titleElement.textContent = currentTitle;
            this.removeSessionRename(sessionId);
            alert(`Failed to rename session: ${error.message}`);
        }
    }

    storeSessionRename(sessionId, newTitle) {
        try {
            const renames = JSON.parse(localStorage.getItem('lattice_session_renames') || '{}');
            renames[sessionId] = newTitle;
            localStorage.setItem('lattice_session_renames', JSON.stringify(renames));
        } catch (error) {
            console.warn('Could not store session rename:', error);
        }
    }

    getSessionRename(sessionId) {
        try {
            const renames = JSON.parse(localStorage.getItem('lattice_session_renames') || '{}');
            return renames[sessionId] || null;
        } catch (error) {
            console.warn('Could not load session rename:', error);
            return null;
        }
    }

    removeSessionRename(sessionId) {
        try {
            const renames = JSON.parse(localStorage.getItem('lattice_session_renames') || '{}');
            delete renames[sessionId];
            localStorage.setItem('lattice_session_renames', JSON.stringify(renames));
        } catch (error) {
            console.warn('Could not remove session rename:', error);
        }
    }

    // ---------------------------------------------------------------------------
    // SESSION INFO DISPLAY
    // ---------------------------------------------------------------------------

    updateSessionInfo(sessionDetails) {
        const sessionInfo = document.getElementById('sessionInfo');
        if (!sessionInfo) return;

        if (sessionDetails && sessionDetails.session_id) {
            const turnCount = sessionDetails.turn_count || 0;
            const lastUpdated = sessionDetails.last_updated ? 
                new Date(sessionDetails.last_updated).toLocaleString() : 'Unknown';
            
            // Use stored rename if available, otherwise use server title
            const storedTitle = this.getSessionRename(sessionDetails.session_id);
            const displayTitle = storedTitle || sessionDetails.title || 'Active Session';
            
            sessionInfo.innerHTML = `
                <div class="session-info-content">
                    <div class="session-info-title">${displayTitle}</div>
                    <div class="session-info-meta">
                        <span>ID: ${sessionDetails.session_id.substring(0, 8)}...</span>
                        <span>Turns: ${turnCount}</span>
                        <span>Updated: ${lastUpdated}</span>
                    </div>
                </div>
            `;
        } else {
            sessionInfo.innerHTML = '<div class="session-info-empty">No active session</div>';
        }
    }

    clearSessionInfo() {
        const sessionInfo = document.getElementById('sessionInfo');
        if (sessionInfo) {
            sessionInfo.innerHTML = '<div class="session-info-empty">No active session</div>';
        }
    }

    updateActiveSessionUI(sessionId) {
        // Remove active class from all session items
        const sessionItems = document.querySelectorAll('.session-item');
        sessionItems.forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to the selected session
        const activeItem = document.querySelector(`[data-session-id="${sessionId}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
    }

    displaySessionError(message) {
        console.error('Session error:', message);
        // Could display a toast notification or update a status area
    }

    // ---------------------------------------------------------------------------
    // SESSION UTILITIES
    // ---------------------------------------------------------------------------

    getActiveSessionId() {
        return this.activeSessionId;
    }

    async getActiveSessionDetails() {
        if (!this.activeSessionId) return null;

        try {
            const response = await fetch(`${this.sessionsApiUrl}/${this.activeSessionId}`);
            if (!response.ok) throw new Error('Failed to fetch session details');
            return await response.json();
        } catch (error) {
            console.error('Error fetching active session details:', error);
            return null;
        }
    }

    async refreshSessionsList() {
        await this.fetchAndRenderSessions();
    }

    // Load messages for a given session
    async loadSessionMessages(sessionId) {
        if (!sessionId) {
            this.clearMessages();
            return;
        }

        try {
            const response = await fetch(`/v1/avatar/sessions/${sessionId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch session details: ${response.status}`);
            }
            const sessionDetails = await response.json();
            
            this.clearMessages();
            sessionDetails.messages.forEach(msg => {
                this.addChatMessage(msg.role, msg.content, false);
            });
        } catch (error) {
            console.error('Error loading session messages:', error);
            this.addChatMessage('daemon', 'Error loading conversation history.');
        }
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SessionManager;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.SessionManager = SessionManager;
}