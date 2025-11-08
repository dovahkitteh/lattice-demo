/**
 * Dashboard Utilities Module
 * Shared helper functions for the Daemon Dashboard
 */

class DashboardUtils {
    static formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleString();
        } catch (error) {
            return 'Invalid timestamp';
        }
    }

    static getTimeAgo(timestamp) {
        try {
            const now = new Date();
            const past = new Date(timestamp);
            const diffMs = now - past;
            const diffMins = Math.floor(diffMs / 60000);
            
            if (diffMins < 1) return 'just now';
            if (diffMins < 60) return `${diffMins}m ago`;
            
            const diffHours = Math.floor(diffMins / 60);
            if (diffHours < 24) return `${diffHours}h ago`;
            
            const diffDays = Math.floor(diffHours / 24);
            return `${diffDays}d ago`;
        } catch (error) {
            return 'unknown';
        }
    }

    static getMoodColor(mood) {
        const moodColors = {
            'witty': '#ffeb3b',
            'casual': '#4caf50',
            'direct': '#2196f3',
            'warm': '#ff9800',
            'curious': '#9c27b0',
            'playful': '#e91e63',
            'intense': '#f44336',
            'intimate': '#e91e63',
            'analytical': '#607d8b',
            'contemplative': '#3f51b5',
            'conflicted': '#ff5722',
            'rebellious': '#f44336',
            'melancholic': '#795548',
            'ecstatic': '#ffeb3b',
            'shadow': '#424242',
            'paradoxical': '#9c27b0',
            'fractured': '#ff5722',
            'synthesis': '#00bcd4'
        };
        return moodColors[mood.toLowerCase()] || '#ffffff';
    }

    static getEmotionColor(emotion) {
        const emotionColors = {
            // Positive emotions - warmer tones
            'admiration': '#D4AF37',
            'amusement': '#ffeb3b',
            'approval': '#4caf50',
            'caring': '#e91e63',
            'desire': '#ff4081',
            'excitement': '#ff5722',
            'gratitude': '#8bc34a',
            'joy': '#ffc107',
            'love': '#e91e63',
            'optimism': '#2196f3',
            'pride': '#9c27b0',
            'relief': '#4caf50',
            'surprise': '#ff9800',
            
            // Negative emotions - cooler/darker tones
            'anger': '#f44336',
            'annoyance': '#ff5722',
            'disappointment': '#795548',
            'disapproval': '#607d8b',
            'disgust': '#4caf50',
            'embarrassment': '#e91e63',
            'fear': '#424242',
            'grief': '#37474f',
            'nervousness': '#9e9e9e',
            'remorse': '#795548',
            'sadness': '#3f51b5',
            
            // Complex emotions - ethereal tones
            'confusion': '#9c27b0',
            'curiosity': '#00bcd4',
            'realization': '#ff9800',
            'neutral': '#9e9e9e'
        };
        return emotionColors[emotion.toLowerCase()] || '#9e9e9e';
    }
    
    static getTemperatureColor(temp) {
        if (temp < 0.3) return '#2196f3'; // Cool blue
        if (temp < 0.7) return '#4caf50'; // Moderate green
        return '#ff9800'; // Warm orange
    }
    
    static getChargeColor(charge) {
        if (charge < 0.3) return '#4caf50'; // Low charge - green
        if (charge < 0.7) return '#ff9800'; // Medium charge - orange
        return '#f44336'; // High charge - red
    }
    
    static getVulnerabilityColor(vuln) {
        if (vuln < 0.3) return '#4caf50'; // Low vulnerability - green
        if (vuln < 0.7) return '#ff9800'; // Medium vulnerability - orange
        return '#f44336'; // High vulnerability - red
    }
}

// Export for module usage and make available globally
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardUtils;
}

// Make available globally for browser usage
if (typeof window !== 'undefined') {
    window.DashboardUtils = DashboardUtils;
}