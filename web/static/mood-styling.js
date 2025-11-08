/**
 * Mood-Specific Styling Module
 * Applies gothic vampire theme mood-specific colors and glows based on daemon state
 */

class MoodStyling {
    constructor() {
        this.moodClasses = {
            // Original rich array of moods
            'intimate': 'mood-intimate',
            'intense': 'mood-intense',
            'contemplative': 'mood-contemplative',
            'curious': 'mood-curious',
            'playful': 'mood-playful',
            'melancholic': 'mood-melancholic',
            'ecstatic': 'mood-ecstatic',
            'rebellious': 'mood-rebellious',
            'analytical': 'mood-analytical',
            'conflicted': 'mood-conflicted',
            'shadow': 'mood-shadow',
            'paradoxical': 'mood-paradoxical',
            'fractured': 'mood-fractured',
            'synthesis': 'mood-synthesis',
            
            // Conversational spectrum moods (legacy)
            'witty': 'mood-witty',
            'casual': 'mood-casual',
            'direct': 'mood-direct',
            'warm': 'mood-warm',
            
            // New adaptive language system - conversational spectrums
            'light': 'spectrum-light',
            'engaged': 'spectrum-engaged',
            'profound': 'spectrum-profound',
            
            // Holistic emotional mood families
            'serene-attunement': 'mood-serene-attunement',
            'creative-reverent-awe': 'mood-creative-reverent-awe',
            'tender-repair': 'mood-tender-repair',
            'joyful-expansion': 'mood-joyful-expansion',
            'catastrophic-abandonment-panic': 'mood-catastrophic-abandonment-panic',
            'collapsed-withdrawal': 'mood-collapsed-withdrawal',
            'manic-ideation-surge': 'mood-manic-ideation-surge'
        };
    }

    /**
     * Apply mood styling to an element based on mood name
     * @param {HTMLElement} element - Element to style
     * @param {string} moodName - Name of the mood
     */
    applyMoodStyling(element, moodName) {
        if (!element || !moodName) return;

        // Clear existing mood classes
        this.clearMoodStyling(element);

        // Normalize mood name by replacing spaces with hyphens
        const normalizedMoodName = moodName.toLowerCase().replace(/\s+/g, '-');
        
        // Apply new mood class if it exists
        const moodClass = this.moodClasses[normalizedMoodName] || this.moodClasses[moodName.toLowerCase()];
        if (moodClass) {
            element.classList.add(moodClass);
        }
    }

    /**
     * Clear all mood styling classes from an element
     * @param {HTMLElement} element - Element to clear styling from
     */
    clearMoodStyling(element) {
        if (!element) return;

        Object.values(this.moodClasses).forEach(className => {
            element.classList.remove(className);
        });
    }

    /**
     * Apply mood styling to emotional state display
     * @param {Object} emotionData - Emotion data with mood information
     */
    styleEmotionalState(emotionData) {
        const container = document.getElementById('emotionalStateContent');
        if (!container) return;

        // Style dominant emotion or mood
        if (emotionData.dominant_emotion) {
            this.applyMoodStyling(container, emotionData.dominant_emotion);
        }

        // Style individual emotion tags
        const emotionTags = container.querySelectorAll('.emotion-tag, .emotion-name');
        emotionTags.forEach(tag => {
            const emotionName = tag.textContent.toLowerCase().trim();
            this.applyMoodStyling(tag, emotionName);
        });
    }

    /**
     * Apply mood styling to current mood display (enhanced for adaptive language system)
     * @param {string|object} moodData - Current daemon mood (string or mood object with dimensions)
     */
    styleCurrentMood(moodData) {
        const container = document.getElementById('currentMoodContent');
        if (!container || !moodData) return;

        let spectrum, descriptors;
        
        if (typeof moodData === 'object' && moodData.current_mood) {
            // Handle data from get_current_mood_state endpoint
            spectrum = moodData.current_mood;
            descriptors = this.generateMoodDescriptors(moodData);
        } else if (typeof moodData === 'string') {
            // Legacy string format
            spectrum = moodData;
            descriptors = [moodData];
        } else {
            // Fallback
            spectrum = 'unknown';
            descriptors = ['unknown'];
        }

        // Apply spectrum styling to container
        this.applyMoodStyling(container, spectrum);

        // Update mood display with both spectrum and descriptors
        this.updateMoodDisplay(container, spectrum, descriptors, moodData);
    }

    /**
     * Generate descriptive mood words from dimensional data
     * @param {object} moodData - Mood data with dimensions
     * @returns {array} Array of descriptive mood words
     */
    generateMoodDescriptors(moodData) {
        if (!moodData || typeof moodData !== 'object') return ['unknown'];

        const dimensions = moodData.mood_dimensions || moodData;
        const descriptors = [];

        // Extract dimensional values (0.0-1.0)
        const lightness = dimensions.lightness || 0.5;
        const engagement = dimensions.engagement || 0.5;
        const profundity = dimensions.profundity || 0.5;
        const warmth = dimensions.warmth || 0.5;
        const intensity = dimensions.intensity || 0.5;

        // Generate primary descriptor based on spectrum and strongest dimensions
        const spectrum = moodData.spectrum || 'unknown';
        
        if (spectrum === 'light') {
            if (lightness > 0.7 && engagement > 0.6) descriptors.push('witty');
            else if (warmth > 0.7) descriptors.push('warm');
            else if (lightness > 0.6) descriptors.push('casual');
            else descriptors.push('direct');
        } else if (spectrum === 'engaged') {
            if (intensity > 0.7) descriptors.push('intense');
            else if (warmth > 0.7) descriptors.push('intimate');
            else if (engagement > 0.7) descriptors.push('curious');
            else if (profundity > 0.5) descriptors.push('analytical');
            else descriptors.push('playful');
        } else if (spectrum === 'profound') {
            if (profundity > 0.8 && intensity > 0.6) descriptors.push('paradoxical');
            else if (profundity > 0.7) descriptors.push('contemplative');
            else if (intensity > 0.7) descriptors.push('conflicted');
            else if (warmth < 0.3) descriptors.push('shadow');
            else if (profundity > 0.6) descriptors.push('melancholic');
            else descriptors.push('synthesis');
        }

        // Add secondary descriptor based on other strong dimensions
        if (intensity > 0.8) descriptors.push('fierce');
        else if (warmth > 0.8) descriptors.push('tender');
        else if (lightness > 0.8) descriptors.push('bright');
        else if (profundity > 0.8) descriptors.push('deep');

        return descriptors.length > 0 ? descriptors : ['unknown'];
    }

    /**
     * Update mood display with spectrum and descriptors
     * @param {HTMLElement} container - Container element
     * @param {string} spectrum - Conversational spectrum
     * @param {array} descriptors - Descriptive mood words
     * @param {object} moodData - Full mood data
     */
    updateMoodDisplay(container, spectrum, descriptors, moodData) {
        // Find or create mood display elements
        let spectrumElement = container.querySelector('.mood-spectrum');
        let descriptorElement = container.querySelector('.mood-descriptors');
        
        if (!spectrumElement) {
            spectrumElement = document.createElement('div');
            spectrumElement.className = 'mood-spectrum';
            container.appendChild(spectrumElement);
        }
        
        if (!descriptorElement) {
            descriptorElement = document.createElement('div');
            descriptorElement.className = 'mood-descriptors';
            container.appendChild(descriptorElement);
        }

        // Update spectrum display
        spectrumElement.textContent = spectrum.charAt(0).toUpperCase() + spectrum.slice(1);
        this.applyMoodStyling(spectrumElement, spectrum);

        // Update descriptors display
        descriptorElement.innerHTML = '';
        descriptors.forEach((descriptor, index) => {
            const descriptorSpan = document.createElement('span');
            descriptorSpan.className = 'mood-descriptor';
            descriptorSpan.textContent = descriptor.charAt(0).toUpperCase() + descriptor.slice(1);
            this.applyMoodStyling(descriptorSpan, descriptor);
            
            if (index > 0) {
                descriptorElement.appendChild(document.createTextNode(' · '));
            }
            descriptorElement.appendChild(descriptorSpan);
        });
    }

    /**
     * Apply mood-based coloring to daemon thoughts
     * @param {string} thoughtType - Type of thought (intense, contemplative, etc.)
     */
    styleDaemonThoughts(thoughtType = 'contemplative') {
        const container = document.getElementById('daemonThoughtsContent');
        if (!container) return;

        this.applyMoodStyling(container, thoughtType);
    }

    /**
     * Initialize mood styling for all relevant elements
     */
    init() {
        // Set up default gothic atmosphere
        this.addGothicAtmosphere();

        // Apply initial mood styling if mood data is available
        this.applyInitialStyling();
    }

    /**
     * Add gothic atmosphere effects to the dashboard
     */
    addGothicAtmosphere() {
        const dashboard = document.querySelector('.dashboard-container');
        if (dashboard) {
            dashboard.style.setProperty('--glow-intensity', '0.8');
        }

        // Add subtle pulsing to status indicators
        const statusDots = document.querySelectorAll('.status-dot');
        statusDots.forEach(dot => {
            if (dot.classList.contains('connected')) {
                dot.style.animation = 'gothic-pulse 2s infinite';
            }
        });
    }

    /**
     * Apply initial styling based on any existing mood data
     */
    applyInitialStyling() {
        // This will be called after initial data load
        // Default to contemplative mood for gothic atmosphere
        setTimeout(() => {
            const moodContent = document.getElementById('currentMoodContent');
            if (moodContent && !moodContent.classList.contains('mood-')) {
                this.applyMoodStyling(moodContent, 'contemplative');
            }
        }, 1000);
    }

    /**
     * Update styling based on new daemon data
     * @param {Object} daemonData - Complete daemon status data
     */
    updateStyling(daemonData) {
        if (!daemonData) return;

        // Update mood styling
        if (daemonData.current_mood) {
            this.styleCurrentMood(daemonData.current_mood);
        }

        // Update emotional state styling
        if (daemonData.emotional_state) {
            this.styleEmotionalState(daemonData.emotional_state);
        }

        // Update thoughts styling based on intensity
        if (daemonData.shadow_charge > 0.7) {
            this.styleDaemonThoughts('intense');
        } else if (daemonData.shadow_charge > 0.4) {
            this.styleDaemonThoughts('conflicted');
        } else {
            this.styleDaemonThoughts('contemplative');
        }
    }
}

// Make MoodStyling globally available
window.MoodStyling = MoodStyling;