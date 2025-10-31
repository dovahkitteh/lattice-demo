// Error boundary component to handle React errors gracefully

import type { FC, ReactNode } from 'react';
import { Component, ErrorInfo } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4 m-2">
          <h3 className="text-red-400 font-semibold mb-2">⚠️ Component Error</h3>
          <p className="text-slate-300 text-sm">
            This component encountered an error. This is likely due to the backend API not being available yet.
          </p>
          <p className="text-slate-400 text-xs mt-2">
            Make sure the Lattice service is running on port 8080.
          </p>
          <button
            onClick={() => {
              this.setState({ hasError: false });
              window.location.reload();
            }}
            className="mt-3 px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm text-white transition-colors"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
