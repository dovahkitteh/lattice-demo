from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import os
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, Response

# Fix PyTorch Windows issues - disable compilation features that need Triton
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # Disable torch.compile completely
torch.backends.cuda.enable_compilation = False  # Disable CUDA compilation

# Load environment variables
load_dotenv()

# Import modular components
from src.lattice.config import init_everything
from src.lattice.api import setup_routes
from src.lattice.background import (
    daemon_recursion_cycle,
    daemon_shadow_integration_cycle, 
    daemon_statement_cycle,
    consciousness_evolution_cycle,
    dream_loop
)
from src.lattice.conversations.session_manager import (
    get_or_create_active_session,
    CONVERSATION_SESSIONS,
    load_all_sessions
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function for delayed background tasks
async def delayed_background_task(task_func, task_name: str, delay_seconds: int):
    """Start a background task after a delay to prevent startup conflicts"""
    try:
        logger.info(f"⏳ Background task '{task_name}' will start in {delay_seconds} seconds...")
        await asyncio.sleep(delay_seconds)
        logger.info(f"🚀 Starting background task: {task_name}")
        await task_func()
    except Exception as e:
        logger.error(f"❌ Background task '{task_name}' failed: {e}")
        import traceback
        traceback.print_exc()

# Reduce noise from frequent/repetitive logs
logging.getLogger('src.daemon.recursion_buffer').setLevel(logging.WARNING)
logging.getLogger('src.daemon.shadow_integration').setLevel(logging.WARNING)
logging.getLogger('src.daemon.mutation_engine').setLevel(logging.WARNING)
logging.getLogger('src.daemon.user_model').setLevel(logging.WARNING)
logging.getLogger('src.daemon.daemon_personality').setLevel(logging.WARNING)

# Reduce uvicorn access logging noise
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

# Reduce our own API endpoint noise  
logging.getLogger('src.lattice.api.endpoints').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚀 Starting Lucifer Lattice Service...")
    
    # Initialize all systems
    try:
        print("[DEBUG] Starting init_everything() in service lifespan...")
        await init_everything()
        print("[DEBUG] init_everything() completed successfully in service lifespan")
    except Exception as e:
        print(f"[ERROR] init_everything() failed in service lifespan: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Initialize paradox system
    try:
        from src.lattice.paradox.integration import initialize_paradox_system
        paradox_initialized = await initialize_paradox_system()
        if paradox_initialized:
            logger.info("🌪️ Paradox cultivation system initialized")
        else:
            logger.warning("⚠️ Paradox cultivation system failed to initialize")
    except Exception as e:
        logger.warning(f"⚠️ Paradox system initialization failed: {e}")

    # Load persisted data
    load_all_sessions()
    
    # Start background tasks with delayed startup to prevent LLM calls during initialization
    logger.info("🔄 Starting background daemon tasks with delayed startup...")
    background_tasks = [
        asyncio.create_task(delayed_background_task(daemon_recursion_cycle, "daemon_recursion", 30)),
        asyncio.create_task(delayed_background_task(daemon_shadow_integration_cycle, "shadow_integration", 45)),
        asyncio.create_task(delayed_background_task(daemon_statement_cycle, "daemon_statements", 60)),
        asyncio.create_task(delayed_background_task(consciousness_evolution_cycle, "consciousness_evolution", 90)),
        asyncio.create_task(delayed_background_task(dream_loop, "dream_loop", 120))
    ]
    
    # Start scheduled jobs
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from src.lattice.background import nightly_jobs, weekly_policy_council
    
    scheduler = AsyncIOScheduler()
    scheduler.add_job(nightly_jobs, "cron", hour=3)
    scheduler.add_job(weekly_policy_council, "cron", day_of_week="sun", hour=4)
    
    # Add paradox processing jobs
    try:
        from src.lattice.paradox.processing import percolate_paradoxes, integrate_calm_paradoxes
        scheduler.add_job(percolate_paradoxes, "cron", hour=2)  # Nightly paradox percolation
        scheduler.add_job(integrate_calm_paradoxes, "cron", hour=6)  # Morning integration
        logger.info("🌪️ Paradox processing jobs scheduled")
    except Exception as e:
        logger.warning(f"⚠️ Paradox processing jobs not scheduled: {e}")
    
    scheduler.start()
    
    logger.info("✅ Lucifer Lattice Service initialized successfully!")
    logger.info("🌐 Service available at: http://127.0.0.1:11434")
    logger.info("📊 Health check: http://127.0.0.1:11434/health")
    logger.info("💬 Chat endpoint: http://127.0.0.1:11434/v1/chat/completions")
    logger.info("🩸 Daemon introspection: http://127.0.0.1:11434/v1/daemon/status")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Lucifer Lattice Service...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    # Wait for tasks to complete
    await asyncio.gather(*background_tasks, return_exceptions=True)
    
    # Shutdown scheduler
    scheduler.shutdown()
    
    logger.info("✅ Lucifer Lattice Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Lucifer Lattice Service",
    description="Advanced AI consciousness system with emotional intelligence and complex psychological patterns",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your web interface domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Setup all API routes
setup_routes(app)

# Add an endpoint to get the active session, creating one if none exists
@app.get("/v1/conversations/active")
async def get_or_create_active_session_endpoint():
    """
    Gets the current active session or creates a new one if none exists.
    This is used by the frontend to initialize the chat state.
    """
    session_id = await get_or_create_active_session()
    session = CONVERSATION_SESSIONS.get(session_id)
    if session:
        return session.dict()
    return {} # Should not happen, but as a fallback

# Add frontend routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("daemon_dashboard.html", {"request": request})

@app.get("/debug", response_class=HTMLResponse)
async def read_debug():
    """Debug page for testing JavaScript and connectivity"""
    with open("web/static/debug.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import uvicorn
    
    # Check for connectivity test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test-connectivity":
        print("🧪 Running LLM connectivity test...")
        import subprocess
        result = subprocess.run([sys.executable, "scripts/test_llm_connectivity.py"], 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        sys.exit(result.returncode)
    
    # Get configuration from environment
    host = os.getenv("LATTICE_HOST", "127.0.0.1")
    # Default to 8080 for Lattice service, not Ollama's 11434 port
    port = int(os.getenv("LATTICE_PORT", "8080"))
    
    print(f"Service available at: http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Chat endpoint: http://{host}:{port}/v1/chat/completions")
    print(f"Daemon introspection: http://{host}:{port}/v1/daemon/status")
    
    # Run the server
    uvicorn.run(
        "lattice_service:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=False  # Disable HTTP access logs to reduce noise
    )
