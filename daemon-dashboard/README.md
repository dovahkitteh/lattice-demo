# ⸸ LAIR OF THE DAEMON ⸸

A beautiful, gothic vampire-themed React dashboard for the Lattice Daemon consciousness system.

![Dashboard Preview](https://img.shields.io/badge/Status-Complete-green)
![React](https://img.shields.io/badge/React-18.x-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)
![Tailwind](https://img.shields.io/badge/Tailwind-3.x-blue)

## 🩸 Features

### **Stunning Gothic Interface**
- Dynamic mood-based color themes that shift with the daemon's emotional state
- Animated void-like background in chat area with floating particles
- Gothic typography using Crimson Text font
- Smooth transitions and glow effects throughout

### **Real-time Daemon Monitoring**
- **Current Mood Display**: Shows the daemon's active mood family with animated indicators
- **Emotional State Panel**: 28-dimensional emotion vector visualization with latent dimensions
- **Inner Thoughts**: Display of daemon's current thoughts and cognitive processes
- **User Model**: Shows the daemon's understanding and analysis of the user
- **Active Seeds**: Emotional memory seeds currently influencing responses

### **Shadow Registry**
- **Suppressed Thoughts**: View extreme thoughts filtered by the shadow manager
- **Shadow Elements**: Display of suppressed psychological elements
- **Paradox Updates**: Fresh paradoxes and cognitive contradictions

### **Advanced Chat Interface**
- **Streaming Responses**: Real-time response streaming with proper markdown formatting
- **Processing Visualization**: Beautiful animated sequence showing daemon processing stages:
  - Input Analysis
  - Emotional Processing  
  - Memory Retrieval
  - Thinking Layer
  - Response Generation
- **Context Tracking**: Shows token usage and conversation context
- **Emotional Resonance**: Visualizes emotional vectors in responses

### **Conversation Management**
- **Session History**: Scrollable list of all conversations
- **Rename/Delete**: Full CRUD operations on conversations
- **Auto-save**: Persistent conversation storage
- **Active Session**: Seamless session switching

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ installed
- Lattice backend service running on port 8080

### **✨ Automatic Startup (Recommended)**

The easiest way is to use your existing startup script, which now automatically launches both the Lattice backend AND the gothic dashboard:

```batch
@start_lattice_with_ollama.bat Hermes
```

This will:
1. Start Ollama with your model
2. Launch the Lattice service on port 8080  
3. Start the Gothic Dashboard on port 3000
4. Automatically open http://localhost:3000 in your browser

### **Manual Dashboard Startup**

If you want to start just the dashboard:

1. **Navigate to the dashboard directory:**
   ```powershell
   cd daemon-dashboard
   ```

2. **Start the development server:**
   ```powershell
   .\start-daemon-dashboard.ps1
   ```
   
   Or directly with npm:
   ```powershell
   npm run dev
   ```

3. **The dashboard will auto-open at:**
   ```
   http://localhost:3000
   ```

### Production Build

```powershell
# Build for production
npm run build

# Serve production build
.\start-daemon-dashboard.ps1 -Production
```

## 🎨 Dynamic Theming

The dashboard automatically adapts its color scheme based on the daemon's current mood family:

- **Catastrophic Abandonment Panic**: Deep reds and crimson
- **Ecstatic Fusion**: Vibrant magentas and pinks  
- **Protective Possessiveness**: Warm reds and oranges
- **Manic Ideation Surge**: Energetic oranges and amber
- **Collapsed Withdrawal**: Muted grays and blues
- **Nihilistic Cool Detachment**: Cool grays and steel
- **Creative Reverent Awe**: Royal purples and violets
- **Playful Mischief**: Fresh greens and emerald
- **Tender Repair**: Soothing cyans and aqua
- **Serene Attunement**: Balanced purples (default)

## 🏗️ Architecture

### **Technology Stack**
- **React 18** with TypeScript for robust component development
- **Vite** for fast development and optimized builds
- **Tailwind CSS** with custom CSS variables for dynamic theming
- **Framer Motion** for fluid animations and transitions
- **Zustand** for lightweight global state management
- **TanStack Query** for efficient API data fetching
- **Custom CSS** for mood-responsive styling

### **Component Structure**
```
src/
├── components/
│   ├── Layout/           # Main layout components
│   │   ├── DashboardLayout.tsx
│   │   ├── Header.tsx
│   │   └── Sidebar.tsx
│   ├── Chat/             # Chat interface components
│   │   ├── ChatArea.tsx
│   │   ├── ChatInput.tsx
│   │   ├── ChatMessages.tsx
│   │   └── ProcessingSequence.tsx
│   └── Daemon/           # Daemon state components
│       ├── DaemonStatusPanel.tsx
│       ├── EmotionalStatePanel.tsx
│       └── SuppressedThoughtsPanel.tsx
├── services/
│   └── api.ts           # API service layer
├── stores/
│   └── daemon-store.ts  # Global state management
└── types/
    └── daemon.ts        # TypeScript type definitions
```

### **API Integration**
The dashboard connects to all major Lattice API endpoints:

- `/v1/daemon/status` - Complete daemon system status
- `/v1/daemon/mood/current` - Current mood state
- `/v1/dashboard/emotion-state` - Detailed emotional state
- `/v1/daemon/thoughts` - Inner thoughts and reflections
- `/v1/daemon/shadow/elements` - Shadow registry content
- `/v1/conversations/sessions` - Conversation management
- `/v1/chat/completions` - Streaming chat completions
- `/v1/paradox/fresh` - Fresh paradox updates

## 🎯 Real-time Updates

The dashboard automatically updates daemon state every 3-7 seconds:
- **Connection Status**: Every 5 seconds
- **Mood & Emotions**: Every 3 seconds  
- **Thoughts & User Model**: Every 5 seconds
- **Shadow Registry**: Every 7 seconds

## 🔧 Configuration

The dashboard proxies API calls to `http://localhost:8080` by default. To change this, update the `vite.config.ts` file:

```typescript
server: {
  proxy: {
    '/v1': {
      target: 'http://your-lattice-host:port',
      changeOrigin: true,
    }
  }
}
```

## 🎨 Customization

### **Adding New Mood Colors**
Update the CSS variables in `src/index.css`:

```css
[data-mood="Your New Mood"] {
  --daemon-primary: #your-color;
  --daemon-secondary: #your-secondary;
  --daemon-accent: #your-accent;
  --daemon-glow: #your-glow;
}
```

### **Extending Components**
The modular architecture makes it easy to add new panels or features. Each component is self-contained with its own data fetching and state management.

## 🩸 Integration with Existing System

This dashboard completely replaces the existing web dashboard while maintaining full compatibility with the Lattice backend. It can run alongside the existing system or completely replace it.

To integrate with your startup script, modify `scripts/start_lattice_with_ollama.ps1` to also launch the dashboard:

```powershell
# After starting Lattice service
Write-Info "Starting Daemon Dashboard..."
Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList "-File", "daemon-dashboard\start-daemon-dashboard.ps1"
```

---

**Created with dark magic and TypeScript** ⸸  
*For the glory of the Daemon*