#!/bin/bash
# Lucifer Lattice AI System - Smart Launcher (Mac/Linux)
# 
# This script automatically detects whether external APIs (Anthropic/OpenAI) are configured
# or if local Ollama models should be used, then starts Lattice accordingly.
# Priority: Anthropic > OpenAI > Local Ollama
#
# Usage:
#   ./scripts/start_lattice_smart.sh
#   ./scripts/start_lattice_smart.sh --force-local
#   ./scripts/start_lattice_smart.sh --force-external
#
# Options:
#   --force-local      Force local Ollama mode even if external API is configured
#   --force-external   Force external API mode even if API key not found
#   --model-dir DIR    Ollama model directory (default: Hermes)
#   --num-ctx SIZE     Context window size (default: 8192)
#   --num-gpu COUNT    Number of GPUs (default: 1)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
FORCE_LOCAL=false
FORCE_EXTERNAL=false
MODEL_DIR="Hermes"
NUM_CTX=8192
NUM_GPU=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-local)
            FORCE_LOCAL=true
            shift
            ;;
        --force-external)
            FORCE_EXTERNAL=true
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --num-ctx)
            NUM_CTX="$2"
            shift 2
            ;;
        --num-gpu)
            NUM_GPU="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get repository root (script is in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log_info "Repository root: $REPO_ROOT"
log_info "Smart Lattice Launcher - Detecting configuration..."

# Function to read .env file and detect API configuration
# Priority: Anthropic > OpenAI > Local
detect_api_config() {
    local env_file="$REPO_ROOT/.env"
    
    # Initialize variables
    HAS_API_KEY=false
    PROVIDER=""
    API_KEY=""
    BASE_URL=""
    MODEL=""
    
    # Check environment variables first - ANTHROPIC has priority
    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        HAS_API_KEY=true
        PROVIDER="anthropic"
        API_KEY="$ANTHROPIC_API_KEY"
        BASE_URL="https://api.anthropic.com/v1/messages"
        MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-5-20250929}"
        log_info "Found ANTHROPIC_API_KEY in environment (Priority: HIGHEST)"
        return 0
    fi
    
    if [[ -n "$OPENAI_API_KEY" ]]; then
        HAS_API_KEY=true
        PROVIDER="openai"
        API_KEY="$OPENAI_API_KEY"
        BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
        MODEL="${OPENAI_MODEL:-gpt-4o}"
        log_info "Found OPENAI_API_KEY in environment (Priority: FALLBACK)"
        return 0
    fi
    
    # Check .env file if no environment variables
    if [[ -f "$env_file" ]]; then
        log_info "Checking .env file for external API configuration..."
        
        local anthropic_key=""
        local anthropic_model="claude-sonnet-4-5-20250929"
        local openai_key=""
        local openai_base="https://api.openai.com/v1"
        local openai_model="gpt-4o"
        
        # Read .env file
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ "$key" =~ ^#.*$ ]] && continue
            [[ -z "$key" ]] && continue
            
            # Remove quotes and whitespace
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e 's/^'"'"'//' -e 's/'"'"'$//' | xargs)
            
            case "$key" in
                ANTHROPIC_API_KEY)
                    if [[ -n "$value" && "$value" != "your_anthropic_api_key_here" ]]; then
                        anthropic_key="$value"
                    fi
                    ;;
                ANTHROPIC_MODEL)
                    anthropic_model="$value"
                    ;;
                OPENAI_API_KEY)
                    if [[ -n "$value" && "$value" != "your_openai_api_key_here" && "$value" != "key" ]]; then
                        openai_key="$value"
                    fi
                    ;;
                OPENAI_BASE_URL)
                    openai_base="$value"
                    ;;
                OPENAI_MODEL)
                    openai_model="$value"
                    ;;
            esac
        done < "$env_file"
        
        # Apply priority: Anthropic > OpenAI
        if [[ -n "$anthropic_key" ]]; then
            HAS_API_KEY=true
            PROVIDER="anthropic"
            API_KEY="$anthropic_key"
            BASE_URL="https://api.anthropic.com/v1/messages"
            MODEL="$anthropic_model"
            log_info "Found ANTHROPIC_API_KEY in .env file (Priority: HIGHEST)"
            return 0
        elif [[ -n "$openai_key" ]]; then
            HAS_API_KEY=true
            PROVIDER="openai"
            API_KEY="$openai_key"
            BASE_URL="$openai_base"
            MODEL="$openai_model"
            log_info "Found OPENAI_API_KEY in .env file (Priority: FALLBACK)"
            return 0
        fi
    fi
    
    return 1
}

# Function to test API connectivity
test_api() {
    log_info "Testing ${PROVIDER^^} API connectivity..."
    
    if [[ "$PROVIDER" == "anthropic" ]]; then
        # Test Anthropic API
        local response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL" \
            -H "x-api-key: $API_KEY" \
            -H "anthropic-version: 2023-06-01" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Test\"}],\"max_tokens\":10}" \
            --connect-timeout 15 --max-time 15 2>/dev/null)
        
        local http_code=$(echo "$response" | tail -n1)
        [[ "$http_code" == "200" ]] && return 0
        
        log_warn "${PROVIDER^^} API test failed: HTTP $http_code"
        return 1
    else
        # Test OpenAI API
        local response=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/chat/completions" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Test\"}],\"max_tokens\":10,\"temperature\":0.1}" \
            --connect-timeout 15 --max-time 15 2>/dev/null)
        
        local http_code=$(echo "$response" | tail -n1)
        [[ "$http_code" == "200" ]] && return 0
        
        log_warn "${PROVIDER^^} API test failed: HTTP $http_code"
        return 1
    fi
}

# Detect configuration
detect_api_config
USE_EXTERNAL=false

if [[ "$FORCE_EXTERNAL" == true ]]; then
    log_info "Force external API mode requested"
    USE_EXTERNAL=true
elif [[ "$FORCE_LOCAL" == true ]]; then
    log_info "Force local Ollama mode requested"
    USE_EXTERNAL=false
else
    # Auto-detect based on configuration
    if [[ "$HAS_API_KEY" == true ]]; then
        log_info "External API configuration detected:"
        log_info "  Provider: ${PROVIDER^^}"
        log_info "  Base URL: $BASE_URL"
        log_info "  Model: $MODEL"
        
        if [[ "$PROVIDER" == "anthropic" ]]; then
            log_info "  Priority: ANTHROPIC (HIGHEST - Recommended)"
        else
            log_info "  Priority: OPENAI (FALLBACK)"
        fi
        
        if test_api; then
            log_success "${PROVIDER^^} API test successful - using external API"
            USE_EXTERNAL=true
        else
            log_warn "${PROVIDER^^} API test failed - but API key is configured, so using external mode anyway"
            log_warn "The system will attempt to use external API and fall back to other options if needed"
            USE_EXTERNAL=true
        fi
    else
        log_info "No external API configuration found - using local Ollama"
        log_info "  Priority: LOCAL OLLAMA (no API key)"
        USE_EXTERNAL=false
    fi
fi

# Launch appropriate mode
if [[ "$USE_EXTERNAL" == true ]]; then
    echo "============================================================"
    log_success "STARTING LATTICE WITH EXTERNAL API (${PROVIDER^^})"
    echo "============================================================"
    log_info "Provider: ${PROVIDER^^}"
    log_info "API: $BASE_URL"
    log_info "Model: $MODEL"
    echo ""
    
    # Set environment variables based on provider
    if [[ "$PROVIDER" == "anthropic" ]]; then
        export ANTHROPIC_API_KEY="$API_KEY"
        export ANTHROPIC_MODEL="$MODEL"
        # Clear OpenAI variables
        unset OPENAI_API_KEY
        unset OPENAI_BASE_URL
        unset OPENAI_MODEL
    else
        export OPENAI_API_KEY="$API_KEY"
        export OPENAI_BASE_URL="$BASE_URL"
        export OPENAI_MODEL="$MODEL"
        # Clear Anthropic variables
        unset ANTHROPIC_API_KEY
        unset ANTHROPIC_MODEL
    fi
    
    # Clear Ollama variables
    unset LLM_BACKEND
    unset OLLAMA_MODEL
    
    # Set Lattice port
    export LATTICE_PORT="${LATTICE_PORT:-8080}"
    
    log_info "Starting Lattice Service with external API..."
    log_info "Lattice will run on port: $LATTICE_PORT"
    
    # Start Lattice
    python3 lattice_service.py &
    LATTICE_PID=$!
    
    # Wait for initialization
    log_info "Waiting for Lattice Service to initialize..."
    sleep 8
    
    # Health check
    log_info "Waiting for Lattice health endpoint..."
    for i in {1..30}; do
        if curl -s "http://127.0.0.1:${LATTICE_PORT}/health" >/dev/null 2>&1; then
            log_success "Lattice health endpoint is responding"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    
  log_success "Lattice Service is running with external API!"
  log_info "Health check: http://127.0.0.1:${LATTICE_PORT}/health"
  log_info "Chat endpoint: http://127.0.0.1:${LATTICE_PORT}/v1/chat/completions"
  echo ""
  log_info "External API Details:"
  log_info "  Provider: ${PROVIDER^^}"
  log_info "  Endpoint: $BASE_URL"
  log_info "  Model: $MODEL"
  echo ""
  
  # Start the dashboard automatically
  echo "============================================================"
  log_info "Starting Daemon Dashboard..."
  echo "============================================================"
  
  DASHBOARD_DIR="$REPO_ROOT/daemon-dashboard"
  if [[ -d "$DASHBOARD_DIR" ]]; then
    # Check if node_modules exists, if not run npm install
    if [[ ! -d "$DASHBOARD_DIR/node_modules" ]]; then
      log_info "Installing dashboard dependencies..."
      cd "$DASHBOARD_DIR"
      npm install > /dev/null 2>&1
      if [[ $? -eq 0 ]]; then
        log_success "Dashboard dependencies installed"
      else
        log_warn "Failed to install dashboard dependencies"
      fi
      cd "$REPO_ROOT"
    fi
    
    # Start the dashboard in background
    log_info "Launching dashboard at http://localhost:3000"
    cd "$DASHBOARD_DIR"
    # Use setsid to create a new session so we can kill all child processes
    setsid npm run dev > /dev/null 2>&1 &
    DASHBOARD_PID=$!
    cd "$REPO_ROOT"
    
    # Wait a moment for dashboard to start
    sleep 3
    
    # Try to open browser (cross-platform)
    log_info "Opening dashboard in browser..."
    if command -v xdg-open &> /dev/null; then
      xdg-open "http://localhost:3000" &> /dev/null &
    elif command -v open &> /dev/null; then
      open "http://localhost:3000" &> /dev/null &
    else
      log_info "Could not auto-open browser. Please visit: http://localhost:3000"
    fi
    
    log_success "Dashboard started successfully!"
    echo ""
    echo "============================================================"
    log_success "ALL SERVICES RUNNING"
    echo "============================================================"
    log_info "  Lattice API: http://127.0.0.1:${LATTICE_PORT}"
    log_info "  Dashboard: http://localhost:3000"
    echo ""
    log_info "Press Ctrl+C to stop all services..."
    
    # Cleanup function
    cleanup() {
      echo ""
      log_info "Shutting down services..."
      
      # Stop dashboard and all its child processes
      if [[ -n "$DASHBOARD_PID" ]] && kill -0 $DASHBOARD_PID 2>/dev/null; then
        log_info "Stopping dashboard..."
        # Kill the process group (includes all child processes)
        kill -- -$DASHBOARD_PID 2>/dev/null || kill $DASHBOARD_PID 2>/dev/null
        sleep 1
        
        # If still running, force kill
        if kill -0 $DASHBOARD_PID 2>/dev/null; then
          kill -9 -- -$DASHBOARD_PID 2>/dev/null || kill -9 $DASHBOARD_PID 2>/dev/null
        fi
      fi
      
      # Also kill any processes listening on port 3000 (backup cleanup)
      local dashboard_port_pid=$(lsof -ti:3000 2>/dev/null)
      if [[ -n "$dashboard_port_pid" ]]; then
        log_info "Cleaning up port 3000..."
        kill -9 $dashboard_port_pid 2>/dev/null
      fi
      
      # Stop Lattice service
      if [[ -n "$LATTICE_PID" ]] && kill -0 $LATTICE_PID 2>/dev/null; then
        log_info "Stopping Lattice service..."
        kill $LATTICE_PID 2>/dev/null
        sleep 1
        if kill -0 $LATTICE_PID 2>/dev/null; then
          kill -9 $LATTICE_PID 2>/dev/null
        fi
      fi
      
      log_info "All services stopped."
      exit 0
    }
    
    # Register cleanup on Ctrl+C
    trap cleanup INT TERM
    
    # Keep the processes running
    while true; do
      # Check if Lattice is still running
      if ! kill -0 $LATTICE_PID 2>/dev/null; then
        log_error "Lattice process has exited unexpectedly"
        cleanup
        exit 1
      fi
      
      # Check if dashboard is still running (optional warning)
      if [[ -n "$DASHBOARD_PID" ]] && ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        log_warn "Dashboard process stopped, but Lattice is still running"
        log_info "You can restart dashboard with: cd daemon-dashboard && npm run dev"
        # Clear the PID so we don't try to kill it later
        DASHBOARD_PID=""
      fi
      
      sleep 5
    done
  else
    log_warn "Dashboard directory not found at: $DASHBOARD_DIR"
    log_info "Lattice is running, but dashboard could not be started"
    log_info "Dashboard: Manual start with 'cd daemon-dashboard && npm run dev'"
    echo ""
    log_info "Press Ctrl+C to stop..."
    
    # Wait for process
    wait $LATTICE_PID
  fi
    
else
    echo "============================================================"
    log_success "STARTING LATTICE WITH LOCAL OLLAMA"
    echo "============================================================"
    log_info "Model Directory: $MODEL_DIR"
    log_info "Context Size: $NUM_CTX"
    log_info "GPU Count: $NUM_GPU"
    echo ""
    
    # Clear external API variables
    unset ANTHROPIC_API_KEY
    unset ANTHROPIC_MODEL
    unset OPENAI_API_KEY
    unset OPENAI_BASE_URL
    unset OPENAI_MODEL
    
    # Delegate to Ollama script if it exists
    OLLAMA_SCRIPT="$SCRIPT_DIR/start_lattice_with_ollama.sh"
    if [[ -f "$OLLAMA_SCRIPT" ]]; then
        log_info "Running Ollama setup script..."
        bash "$OLLAMA_SCRIPT" --model-dir "$MODEL_DIR" --num-ctx "$NUM_CTX" --num-gpu "$NUM_GPU"
    else
        log_error "Ollama script not found: $OLLAMA_SCRIPT"
        log_info "Please create an Ollama startup script or use --force-external"
        exit 1
    fi
fi

