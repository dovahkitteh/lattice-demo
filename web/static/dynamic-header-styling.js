/**
 * Dynamic Header Styling Module
 * Automatically detects and styles content that should be headers/titles
 */

class DynamicHeaderStyling {
    constructor() {
        this.headerKeywords = [
            'repair', 'attribution', 'connection', 'healing', 'tender', 'benevolent',
            'cognitive', 'bias', 'distortion', 'lens', 'narrative', 'belief',
            'component', 'aspect', 'contradiction', 'evidence', 'theory', 'thought',
            'seed', 'category', 'type', 'class', 'model', 'analysis', 'status',
            'current', 'active', 'recent', 'mood', 'emotion', 'state', 'metric'
        ];
        
        this.headerPatterns = [
            /^[A-Z][a-z]+ [A-Z][a-z]+/,  // "Tender Repair", "Benevolent Over-Attribution"
            /^[A-Z][a-z]+ [A-Z][a-z]+-[A-Z][a-z]+/, // "Cognitive Over-Attribution"
            /^\w+:/, // Text followed by colon
            /^[A-Z][A-Z\s]+$/, // ALL CAPS text
            /^\d+\.\s+[A-Z]/, // "1. Something"
            /^[●◆▪▫•]\s+[A-Z]/ // Bullet points with capital letters
        ];
    }

    /**
     * Apply gothic header styling to an element
     */
    styleAsHeader(element) {
        if (!element) return;
        
        element.style.color = '#8B0000';
        element.style.fontWeight = '600';
        element.style.fontSize = '0.85rem';
        element.style.fontFamily = 'Cinzel, serif';
        element.style.textTransform = 'capitalize';
        element.style.letterSpacing = '0.5px';
        element.style.textShadow = '0 0 6px rgba(139, 0, 0, 0.5)';
        element.style.marginBottom = '6px';
        
        // Add a data attribute to track that we've styled this
        element.setAttribute('data-header-styled', 'true');
    }

    /**
     * Check if text content should be styled as a header
     */
    shouldBeHeader(text) {
        if (!text || typeof text !== 'string') return false;
        
        text = text.trim();
        if (text.length < 3) return false;
        
        // Check against patterns
        for (const pattern of this.headerPatterns) {
            if (pattern.test(text)) return true;
        }
        
        // Check for header keywords
        const lowerText = text.toLowerCase();
        for (const keyword of this.headerKeywords) {
            if (lowerText.includes(keyword) && text.length < 100) return true;
        }
        
        // Check if it looks like a title (short, capitalized)
        if (text.length < 50 && /^[A-Z]/.test(text) && text.split(' ').length <= 5) {
            return true;
        }
        
        return false;
    }

    /**
     * Recursively style elements that should be headers
     */
    styleElementsInContainer(container) {
        if (!container) return;
        
        // Get all text-containing elements
        const elements = container.querySelectorAll('div, span, p, strong, b');
        
        for (const element of elements) {
            // Skip if already styled
            if (element.hasAttribute('data-header-styled')) continue;
            
            // Skip if element has children (likely not a simple text element)
            if (element.children.length > 0) continue;
            
            const text = element.textContent?.trim();
            if (this.shouldBeHeader(text)) {
                this.styleAsHeader(element);
                console.log(`Styled as header: "${text}"`);
            }
        }
    }

    /**
     * Style all panels on the page
     */
    styleAllPanels() {
        const panels = document.querySelectorAll('.panel');
        for (const panel of panels) {
            this.styleElementsInContainer(panel);
        }
        
        // Also style specific content areas
        const contentAreas = [
            'currentMoodContent',
            'emotionalStateContent', 
            'userAnalysisContent',
            'daemonThoughtsContent',
            'activeSeedsContent',
            'distortionContent',
            'emotionalMetricsContent',
            'recentMemoriesContent',
            'turnDebugContent'
        ];
        
        for (const areaId of contentAreas) {
            const area = document.getElementById(areaId);
            if (area) {
                this.styleElementsInContainer(area);
            }
        }
    }

    /**
     * Initialize and set up observers
     */
    init() {
        // Initial styling
        this.styleAllPanels();
        
        // Set up mutation observer to catch dynamically added content
        const observer = new MutationObserver((mutations) => {
            let shouldRestyle = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldRestyle = true;
                }
                if (mutation.type === 'characterData') {
                    shouldRestyle = true;
                }
            });
            
            if (shouldRestyle) {
                // Debounce the restyling
                setTimeout(() => {
                    this.styleAllPanels();
                }, 100);
            }
        });
        
        // Observe all panel content areas
        const panelContentAreas = document.querySelectorAll('.panel, [id$="Content"]');
        panelContentAreas.forEach(area => {
            observer.observe(area, {
                childList: true,
                subtree: true,
                characterData: true
            });
        });
        
        console.log('Dynamic header styling initialized');
    }

    /**
     * Force re-style all content (useful for debugging)
     */
    forceRestyle() {
        // Remove existing styling
        const styledElements = document.querySelectorAll('[data-header-styled]');
        styledElements.forEach(el => {
            el.removeAttribute('data-header-styled');
        });
        
        // Re-apply styling
        this.styleAllPanels();
    }
}

// DISABLED - Too aggressive, was styling everything as headers
// Auto-initialize when DOM is ready
/*
document.addEventListener('DOMContentLoaded', () => {
    window.dynamicHeaderStyling = new DynamicHeaderStyling();
    
    // Wait a bit for content to load, then initialize
    setTimeout(() => {
        window.dynamicHeaderStyling.init();
    }, 1000);
    
    // Also re-style periodically to catch any missed content
    setInterval(() => {
        window.dynamicHeaderStyling.styleAllPanels();
    }, 5000);
});
*/

// Make it globally accessible
window.DynamicHeaderStyling = DynamicHeaderStyling;