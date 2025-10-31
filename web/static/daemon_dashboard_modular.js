/**
 * ▲ GLASSSHARD DAEMONCORE Dashboard ▲ - Modular Architecture
 * Real-time monitoring and control for recursive sentience evolution
 * Gothic Vampire Theme - Where shadows meet consciousness
 * 
 * This is the new modular version of the dashboard that replaces the monolithic daemon_dashboard.js
 * 
 * Module Structure:
 * - DashboardUtils: Shared utilities (colors, time formatting, etc.)
 * - DashboardDataService: API communication layer
 * - DashboardComponents: Memory, emotion, and personality displays
 * - SessionManager: Session CRUD operations and management
 * - ChatInterface: Real-time messaging with streaming support
 * - EmotionalSystem: Advanced emotional analysis and mood tracking
 * - DebugTools: Development utilities and data export
 * - DashboardCore: Main orchestration and initialization
 */

// Ensure all required modules are loaded before initializing
(function() {
    'use strict';
    
    // Check if all required classes are available
    const requiredClasses = [
        'DashboardUtils',
        'DashboardDataService', 
        'DashboardComponents',
        'SessionManager',
        'ChatInterface',
        'EmotionalSystem',
        'DebugTools',
        'DashboardCore'
    ];
    
    function checkModulesLoaded() {
        const missingModules = requiredClasses.filter(className => typeof window[className] === 'undefined');
        
        if (missingModules.length > 0) {
            console.warn('Missing required modules:', missingModules);
            console.warn('Please ensure all module scripts are loaded before this file');
            return false;
        }
        
        return true;
    }
    
    function initializeDashboard() {
        if (checkModulesLoaded()) {
            console.log('✅ All modules loaded successfully');
            console.log('🚀 Initializing modular Daemon Dashboard...');
            
            // Initialize the dashboard
            window.app = new DashboardCore();
            
            console.log('🎯 Modular Daemon Dashboard initialized');
        } else {
            console.error('❌ Failed to initialize dashboard: Missing required modules');
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    } else {
        // DOM is already loaded
        initializeDashboard();
    }
    
    // Export version info
    window.DASHBOARD_VERSION = {
        version: '2.0.0-modular',
        architecture: 'modular',
        modules: requiredClasses.length,
        buildDate: new Date().toISOString(),
        features: [
            'Modular Architecture',
            'Real-time Streaming Chat',
            'Advanced Emotional Analysis',
            'Mood Tracking & User Analysis',
            'Session Management',
            'Debug Tools & Data Export',
            'Gothic Vampire Theme',
            'Daemon Consciousness Monitoring'
        ]
    };
    
})();

// Legacy compatibility layer for any external code that might reference the old class
if (typeof window !== 'undefined') {
    // Provide backward compatibility
    window.DaemonDashboard = window.DashboardCore;
}