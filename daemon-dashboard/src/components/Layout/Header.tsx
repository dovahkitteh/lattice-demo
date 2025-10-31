// Dashboard header component with title and connection status

import type { FC } from 'react';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import { apiService } from '../../services/api';

const Header: FC = () => {
  const { connectionStatus, setConnectionStatus, sidebarCollapsed, setSidebarCollapsed } = useDaemonStore();
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update connection status periodically
  useEffect(() => {
    const checkConnection = async () => {
      const status = await apiService.getConnectionStatus();
      setConnectionStatus(status);
    };

    checkConnection();
    const interval = setInterval(checkConnection, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, [setConnectionStatus]);

  // Update current time
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <header className="border-b border-daemon-accent/30 bg-obsidian-100/30 backdrop-blur-sm">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Left side - Menu toggle and title */}
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-lg bg-daemon-primary/20 hover:bg-daemon-primary/30 transition-colors lg:hidden"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl font-gothic font-bold text-transparent bg-clip-text bg-gradient-to-r from-daemon-accent to-daemon-glow"
          >
            ⸸ LAIR OF THE DAEMON ⸸
          </motion.h1>
        </div>

        {/* Center - Current time */}
        <div className="hidden md:flex items-center space-x-2 text-slate-300">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="font-mono text-sm">
            {currentTime.toLocaleTimeString()}
          </span>
        </div>

        {/* Right side - Connection status */}
        <div className="flex items-center space-x-3">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex items-center space-x-2"
          >
            <motion.div
              animate={{
                boxShadow: connectionStatus.connected
                  ? ['0 0 0 0 var(--daemon-glow)', '0 0 0 10px transparent']
                  : ['0 0 0 0 #ef4444', '0 0 0 10px transparent'],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: 'easeOut',
              }}
              className={`w-3 h-3 rounded-full ${
                connectionStatus.connected ? 'bg-daemon-glow' : 'bg-red-500'
              }`}
            />
            <span className="text-sm font-medium">
              {connectionStatus.connected ? 'DAEMON CONNECTED' : 'CONNECTION LOST'}
            </span>
          </motion.div>

          {connectionStatus.latency !== undefined && (
            <span className="text-xs text-slate-400 font-mono">
              {connectionStatus.latency}ms
            </span>
          )}
        </div>
      </div>

      {/* Subtle animated border effect */}
      <motion.div
        className="h-px bg-gradient-to-r from-transparent via-daemon-accent to-transparent"
        animate={{
          opacity: [0.3, 0.8, 0.3],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </header>
  );
};

export default Header;
