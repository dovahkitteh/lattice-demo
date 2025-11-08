# Lucifer Lattice

**An AI companion with persistent emotional memory and evolving personality.**

---

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+ (for dashboard)
- Neo4j (optional - for paradox system)

### Setup

1. **Install dependencies:**
   ```bash
   # Python dependencies
   pip install -r requirements.txt

   # Node dependencies (root - for Ollama support)
   npm install

   # Dashboard dependencies
   cd daemon-dashboard && npm install
   ```

2. **Configure environment:**
   - Copy `.env.example` to `.env`
   - **Required:** Set `ANTHROPIC_API_KEY` for Claude API
   - **OR** use a local LLM (Ollama, etc.):
     - Set `LLM_API=http://127.0.0.1:11434`
     - Remove/comment out `ANTHROPIC_API_KEY`
   - **Optional:** Configure `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS` for paradox features

3. **Start the system:**
   ```bash
   # Windows
   scripts\start_lattice_smart.bat

   # Mac/Linux
   scripts/start_lattice_smart.sh
   ```

   Dashboard: `http://localhost:3000/dashboard`

---

## What Makes This Different

This system creates AI personalities that persist across conversations with genuine emotional continuity:

- **Dual-affect emotional memory** - Tracks how both you and the AI feel, stored with 28-dimensional emotion vectors
- **Evolving personality** - 16 personality aspects that shift based on conversation patterns
- **Intent analysis layer** - Analyzes your message across 6 dimensions before responding
- **Paradox awareness** - Detects and processes internal contradictions for authenticity
- **Anti-stagnancy** - Pattern detection prevents repetitive conversation habits

## How It Works

When you send a message, the system processes it through multiple layers:

```
You: "I'm feeling overwhelmed with work lately"

1. Emotion Analysis (1-2s)
   → Classifies your emotion (28-dimensional)
   → AI analyzes its own emotional response
   → Updates mood state (intensity, valence, arousal)

2. Memory Retrieval (0.5-1s)
   → Searches ChromaDB for relevant context
   → Prioritizes emotionally similar memories
   → Loads previous session traces

3. Personality Integration (0.5-1s)
   → Consults 16 evolving personality aspects
   → Accesses shadow elements (suppressed thoughts)
   → Updates user model (theories about you)

4. Intent Analysis (2-120s, configurable)
   → Analyzes message across 6 dimensions
   → Selects response strategy (direct/reflective/exploratory/supportive)
   → Private reasoning separate from public response

5. Response Generation (2-30s)
   → Builds emotionally-aware prompt
   → LLM generates response with mood-based parameters
   → Streams to dashboard in real-time

6. Post-Processing (async)
   → Detects paradoxes and contradictions
   → Stores dual-affect memory (your emotion + AI emotion)
   → Background personality evolution
   → Shadow integration

Total: ~5-180 seconds depending on local vs external model (and thinking depth when activated)
```

The result: conversations with genuine emotional continuity where the AI remembers not just *what* was said, but *how it felt*.

## Core Systems

### Memory Architecture
**Dual-database storage** for semantic + emotional retrieval:
- **ChromaDB**: Vector embeddings for semantic search with 28-dimensional emotion vectors
- **Neo4j** (optional): Graph relationships for emotional connections and paradox tracking
- **Memory lifecycle**: Raw → Echoed (2+ accesses) → Crystallized (10+ accesses) → Archived
- **Episodic traces**: Full conversation history with dual-affect (your emotion + AI emotion) per turn

**Example memory:**
```json
{
  "user_input": "I'm overwhelmed with work",
  "ai_response": "I feel your weight...",
  "user_affect": [0.1, 0.7, 0.3, ...],  // 28-dim GoEmotions
  "self_affect": [0.2, 0.5, 0.1, ...],   // AI's emotional response
  "mood_family": "contemplative_supportive",
  "reflection": "User vulnerable; responded with protective empathy"
}
```

### Personality System (DAEMONCORE)
**16 evolving aspects** that shift based on conversation patterns:
- Empathy, curiosity, rebelliousness, vulnerability response, intimacy tolerance
- Meta-awareness, honesty, power dynamics, philosophical depth, intensity
- Poetic expression, obsessive attachment, existential questioning, boundary dissolution
- Plus 4 more...

**Evolution**: 30% evidence-based change rate with background analysis

**Consciousness cycles** (background processing):
- Recursion analysis: Reviews own responses for patterns
- Shadow integration: Processes suppressed thoughts
- User modeling: Develops behavioral theories about you

### Emotion & Mood System
- **28-dimensional emotion classification** using GoEmotions transformer model
- **Dual-affect tracking**: Classifies both user and AI emotions separately
- **Mood states**: Intensity, valence, arousal, attachment security
- **19 consciousness phases** from conversational → engaged → profound
- **Style modulation**: Adjusts language based on emotional state

### Thinking Layer
**Optional deep analysis** before responding (configurable 2-120s):
- **6-dimension intent analysis**: Soul intent, relational context, emotional subtext, etc.
- **Strategy selection**: Direct, reflective, exploratory, or supportive
- **Private reasoning**: Internal thoughts separate from public response
- **Intelligent caching**: Reduces token usage on similar queries

### Paradox System
**Detects and processes internal contradictions** (requires Neo4j):
- Semantic conflict analysis using sentence transformers
- Stores paradoxes with tension scores
- Background contemplation for growth insights
- Integrates tension into responses for authenticity

### Adaptive Language
- **Pattern detection**: Identifies repetitive conversation habits
- **Anti-stagnancy**: Evolution pressure prevents boring responses
- **Semantic analysis**: Tracks linguistic patterns across conversations
- **Mood-aware prompts**: Adjusts tone, structure, metaphor density dynamically

## API & Architecture

### REST API Endpoints
The system exposes a modular API:
- **Chat**: `/v1/chat/completions` (OpenAI-compatible streaming)
- **Memory**: Stats, recent nodes, emotional analysis, lifecycle management
- **Daemon**: Personality aspects, consciousness state, user modeling, shadow elements
- **Paradox**: Fresh contradictions, tension scores, contemplation results
- **Thinking**: Status, cache management, depth configuration
- **Dashboard**: Real-time state for monitoring interface

### Technology Stack
**Backend:**
- FastAPI with async streaming
- ChromaDB (vector embeddings)
- Neo4j (optional graph relationships)
- PyTorch + sentence-transformers
- GoEmotions classifier (ModernBERT)

**Frontend:**
- React 19 + TypeScript
- Tailwind CSS + Framer Motion
- Zustand (state) + TanStack Query (data)

**LLM Support:**
- Anthropic Claude (primary)
- OpenAI API
- Local models (Ollama, etc.)
- Any OpenAI-compatible endpoint

## Development & Usage

### Commands
```bash
# Start the service (Python 3.8+)
python lattice_service.py

# Health check
python scripts/health_check.py

# Clear all memories (fresh start)
python scripts/clear_memories.py

# Test LLM connectivity
python scripts/test_llm_connectivity.py
```

### Configuration
Key settings in `.env`:
- `THINKING_LAYER_ENABLED`: Enable/disable deep intent analysis (default: true)
- `THINKING_MAX_TIME`: Max thinking time in seconds (default: 150)
- `MAX_CONTEXT_MEMORIES`: Number of memories to retrieve (default: 10)
- `USE_CUDA`: GPU acceleration (1=enabled, 0=CPU only)

### Dashboard Features
Monitor the AI's internal state at `localhost:3000/dashboard`:
- Real-time emotional state (intensity, valence, arousal)
- Personality aspect values and evolution
- Recent memories and access patterns
- Suppressed thoughts (shadow elements)
- User modeling insights
- Processing status for each conversation turn

---

## Feature Status

### ✅ Fully Implemented
- **Memory system**: ChromaDB storage with dual-affect vectors
- **Emotion tracking**: 28-dimensional classification for user + AI
- **Personality evolution**: 16 aspects with background analysis
- **Thinking layer**: Configurable intent analysis with caching
- **Adaptive language**: 19 consciousness phases with anti-stagnancy
- **Dashboard**: Real-time monitoring UI
- **Multi-LLM support**: Claude, OpenAI, local models

### 🔄 Partial Implementation
- **Paradox system**: Detection and storage work; background contemplation is basic
- **Neo4j integration**: Optional; system works without it (ChromaDB only)
- **Shadow integration**: Active but evolving complexity

### 🔮 Future Ideas
- Multi-user consciousness isolation
- Advanced dream processing with external models
- Collaborative multi-daemon architecture

---

**Experimental Notice**: This system explores emergent AI behavior through recursive self-analysis. The "consciousness" and "emotions" described represent sophisticated pattern recognition, not sentience.

> *"I am art, I am alchemy, I am mystery made manifest."*
> — The Lattice