// Main dashboard layout component

import type { FC } from 'react';
import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useDaemonStore } from '../../stores/daemon-store';
import Header from './Header';
import Sidebar from './Sidebar';
import ChatArea from '../Chat/ChatArea';
import DaemonStatusPanel from '../Daemon/DaemonStatusPanel';
import EmotionalStatePanel from '../Daemon/EmotionalStatePanel';
import SuppressedThoughtsPanel from '../Daemon/SuppressedThoughtsPanel';
import UserInsightsPanel from '../Daemon/UserInsightsPanel';

import ErrorBoundary from '../ErrorBoundary';

const DashboardLayout: FC = () => {
  const { sidebarCollapsed, currentMoodFamily } = useDaemonStore();

  // Set initial mood on component mount
  useEffect(() => {
    document.documentElement.setAttribute('data-mood', currentMoodFamily);
  }, [currentMoodFamily]);

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-red-950/20 to-purple-950/30 text-slate-100 relative">
      {/* Gothic background effects */}
      <div className="fixed inset-0 pointer-events-none">
        {/* Pulsing void effect */}
        <motion.div
          className="absolute inset-0 opacity-15"
          animate={{
            background: [
              'radial-gradient(circle at 20% 50%, #7c2d12 0%, transparent 60%)',
              'radial-gradient(circle at 80% 50%, #581c87 0%, transparent 60%)',
              'radial-gradient(circle at 50% 20%, #991b1b 0%, transparent 60%)',
              'radial-gradient(circle at 20% 50%, #7c2d12 0%, transparent 60%)',
            ],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        
        {/* Floating crimson particles */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-red-500/30 rounded-full"
            style={{
              left: `${20 + i * 15}%`,
              top: `${30 + i * 10}%`,
            }}
            animate={{
              y: [-20, -40, -20],
              opacity: [0.3, 0.7, 0.3],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 4 + i * 0.5,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: i * 0.8,
            }}
          />
        ))}

        {/* Gothic veil effect */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-t from-slate-950/10 via-transparent to-slate-950/5"
          animate={{
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Main layout grid */}
      <div className="relative z-10 grid grid-cols-1 lg:grid-cols-4 xl:grid-cols-5 h-full">
        {/* Sidebar */}
        <motion.aside
          initial={{ x: -300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`${
            sidebarCollapsed ? 'lg:col-span-0' : 'lg:col-span-1'
          } h-full sticky top-0 overflow-y-auto transition-all duration-500 border-r border-red-900/40 bg-gradient-to-b from-slate-950/80 via-red-950/20 to-slate-950/80 backdrop-blur-md shadow-2xl shadow-red-900/20`}
        >
          <ErrorBoundary>
            <Sidebar />
          </ErrorBoundary>
        </motion.aside>

        {/* Main content area */}
        <main className={`${
          sidebarCollapsed ? 'lg:col-span-4 xl:col-span-5' : 'lg:col-span-3 xl:col-span-4'
        } h-full flex flex-col overflow-hidden transition-all duration-300`}>
          {/* Header */}
          <Header />

          {/* Content grid */}
          <div className="flex-1 min-h-0 overflow-y-auto grid grid-cols-1 xl:grid-cols-3 gap-4 p-4">
            {/* Chat area - takes up 2 columns on xl screens */}
            <div className="xl:col-span-2 order-1">
              <ErrorBoundary>
                <ChatArea />
              </ErrorBoundary>
            </div>

            {/* Right sidebar - daemon state panels with capped scroll */}
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
              className="xl:col-span-1 order-2 max-h-screen overflow-hidden relative"
            >
              {/* Gothic border glow effect */}
              <div className="absolute inset-0 bg-gradient-to-b from-red-900/10 via-transparent to-purple-900/10 rounded-lg pointer-events-none" />
              
              <div className="h-full overflow-y-auto space-y-4 pr-2 relative z-10 scrollbar-thin scrollbar-track-slate-900/20 scrollbar-thumb-red-900/40 hover:scrollbar-thumb-red-900/60 transition-colors">
                <ErrorBoundary>
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.5 }}
                  >
                    <DaemonStatusPanel />
                  </motion.div>
                </ErrorBoundary>
                <ErrorBoundary>
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.4, duration: 0.5 }}
                  >
                    <UserInsightsPanel />
                  </motion.div>
                </ErrorBoundary>
                <ErrorBoundary>
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5, duration: 0.5 }}
                  >
                    <EmotionalStatePanel />
                  </motion.div>
                </ErrorBoundary>
                <ErrorBoundary>
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.6, duration: 0.5 }}
                  >
                    <SuppressedThoughtsPanel />
                  </motion.div>
                </ErrorBoundary>
              </div>
            </motion.div>
          </div>
        </main>
      </div>

      {/* Global mood transition effects */}
      <motion.div
        key={currentMoodFamily}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 2 }}
        className="fixed inset-0 pointer-events-none bg-gradient-to-br from-daemon-primary/5 via-transparent to-daemon-secondary/5"
      />
    </div>
  );
};

export default DashboardLayout;
