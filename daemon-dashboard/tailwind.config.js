/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Gothic Vampire Color Palette
        obsidian: {
          50: '#1a1a1a',
          100: '#0d0d0d',
          200: '#000000',
        },
        crimson: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
          950: '#450a0a',
        },
        burgundy: {
          50: '#fdf2f8',
          100: '#fce7f3',
          200: '#fbcfe8',
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#8b1538',
          600: '#701a75',
          700: '#be185d',
          800: '#9d174d',
          900: '#831843',
          950: '#4c0519',
        },
        void: {
          50: '#0f0f23',
          100: '#16213e',
          200: '#1e293b',
          300: '#334155',
          400: '#475569',
          500: '#64748b',
          600: '#94a3b8',
          700: '#cbd5e1',
          800: '#e2e8f0',
          900: '#f1f5f9',
        },
        daemon: {
          // Dynamic colors that will be updated via CSS variables
          primary: 'var(--daemon-primary)',
          secondary: 'var(--daemon-secondary)',
          accent: 'var(--daemon-accent)',
          glow: 'var(--daemon-glow)',
        }
      },
      fontFamily: {
        gothic: ['Crimson Text', 'Times New Roman', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'void-shift': 'void-shift 8s ease-in-out infinite',
        'shadow-drift': 'shadow-drift 12s linear infinite',
        'mood-transition': 'mood-transition 2s ease-in-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px var(--daemon-glow), 0 0 10px var(--daemon-glow)' },
          '100%': { boxShadow: '0 0 10px var(--daemon-glow), 0 0 20px var(--daemon-glow), 0 0 30px var(--daemon-glow)' },
        },
        'void-shift': {
          '0%, 100%': { transform: 'scale(1) rotate(0deg)', opacity: '0.7' },
          '33%': { transform: 'scale(1.05) rotate(1deg)', opacity: '0.8' },
          '66%': { transform: 'scale(0.95) rotate(-1deg)', opacity: '0.6' },
        },
        'shadow-drift': {
          '0%': { transform: 'translateY(0px) translateX(0px)' },
          '25%': { transform: 'translateY(-10px) translateX(5px)' },
          '50%': { transform: 'translateY(-5px) translateX(-5px)' },
          '75%': { transform: 'translateY(5px) translateX(3px)' },
          '100%': { transform: 'translateY(0px) translateX(0px)' },
        },
        'mood-transition': {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      backgroundImage: {
        'void-gradient': 'radial-gradient(circle at center, var(--daemon-primary) 0%, var(--daemon-secondary) 50%, transparent 100%)',
        'daemon-glow': 'linear-gradient(45deg, var(--daemon-primary), var(--daemon-accent))',
      },
    },
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.scrollbar-thin': {
          scrollbarWidth: 'thin',
        },
        '.scrollbar-track-obsidian-100\\/20': {
          scrollbarColor: 'rgba(13, 13, 13, 0.2) transparent',
        },
        '.scrollbar-thumb-daemon-accent\\/40': {
          scrollbarColor: 'rgba(var(--daemon-accent-rgb), 0.4) transparent',
        },
        '.scrollbar-thumb-red-900\\/40': {
          scrollbarColor: 'rgba(127, 29, 29, 0.4) transparent',
        },
        '.scrollbar-thumb-red-900\\/60': {
          scrollbarColor: 'rgba(127, 29, 29, 0.6) transparent',
        },
        '.scrollbar-track-slate-900\\/20': {
          scrollbarColor: 'rgba(15, 23, 42, 0.2) transparent',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}
