import os
import torch
import logging
from typing import Optional, Any, List, Dict
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timezone

from .daemon_setup import init_daemon_core

# Load environment variables
load_dotenv()

# Suppress sentence transformers and tokenizer progress bars globally
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix PyTorch Windows issues - disable compilation features that need Triton
os.environ["TORCH_LOGS"] = ""  # Disable verbose logs
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True  # Disable torch.compile completely
torch.backends.cuda.enable_compilation = False  # Disable CUDA compilation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce daemon system logging to prevent spam
logging.getLogger('src.daemon.recursion_buffer').setLevel(logging.WARNING)
logging.getLogger('src.daemon.shadow_integration').setLevel(logging.WARNING)
logging.getLogger('src.daemon.mutation_engine').setLevel(logging.WARNING)
# Enable debug logging for user model to debug the issue
logging.getLogger('src.daemon.user_model').setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------

LATTICE_DB_PATH = os.getenv("LATTICE_DB_PATH", "./data/lattice")
EMBED_DIM = 768              # embedding size for nomic-embed-text-v1
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "16384"))  # Updated for Hermes 3
RECURSION_BUFFER_SIZE = int(os.getenv("RECURSION_BUFFER_SIZE", "13"))
COMPRESSION_TRIGGER_RATIO = float(os.getenv("COMPRESSION_TRIGGER_RATIO", "0.8"))
POLICY_SIGNING_KEY = os.getenv("POLICY_SIGNING_KEY")
TEST_MODE = os.getenv("LATTICE_TEST_MODE", "false").lower() == "true"

# Dashboard state cache for real-time data access
DASHBOARD_STATE_CACHE = {}

# Thinking Layer Configuration
THINKING_LAYER_ENABLED = os.getenv("THINKING_LAYER_ENABLED", "true").lower() == "true"
THINKING_MAX_TIME = float(os.getenv("THINKING_MAX_TIME", "200.0"))
THINKING_DEPTH_THRESHOLD = float(os.getenv("THINKING_DEPTH_THRESHOLD", "0.6"))
THINKING_DEBUG_LOGGING = os.getenv("THINKING_DEBUG_LOGGING", "false").lower() == "true"

# API Configuration
# Default to local lattice service; can be overridden via LLM_API
TEXT_GENERATION_API_URL = os.getenv("LLM_API", "http://127.0.0.1:8080")
LLM_API_URL = TEXT_GENERATION_API_URL  # Alias for compatibility
STREAM_DELAY = 0.05  # Delay between streaming chunks

# CONVERSATION MANAGEMENT globals
CONVERSATION_SESSIONS = {}  # session_id -> ConversationSession
ACTIVE_SESSION_ID = None
MAX_CONVERSATION_TOKENS = 10000  # Increased for Hermes 3's larger context
COMPRESSION_THRESHOLD = 0.85  # Start compression at 85% of context window

# ---------------------------------------------------------------------------
# DEVICE DETECTION
# ---------------------------------------------------------------------------

def detect_device():
    """Detect device, with clean fallback to CPU on any GPU issues"""
    try:
        use_cuda = os.getenv("USE_CUDA", "1") == "1"
        
        if use_cuda and torch.cuda.is_available():
            # Test GPU with a simple operation to make sure it actually works
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1  # Simple operation test
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
                return "cuda"
            except Exception as gpu_error:
                print(f"GPU test failed ({gpu_error}), falling back to CPU")
                return "cpu"
        else:
            if use_cuda:
                print("CUDA requested but not available, using CPU")
            else:
                print("Using CPU (CUDA disabled)")
            return "cpu"
    except Exception as e:
        print(f"Device detection error: {e}, using CPU")
        return "cpu"

DEVICE = detect_device()

# ---------------------------------------------------------------------------
# GPU MEMORY MANAGEMENT
# ---------------------------------------------------------------------------

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            return {"allocated_gb": allocated, "cached_gb": cached}
        return {"allocated_gb": 0, "cached_gb": 0}
    except:
        return {"allocated_gb": 0, "cached_gb": 0}

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
    except:
        pass
    return False

# ---------------------------------------------------------------------------
# GLOBAL SYSTEM INSTANCES
# ---------------------------------------------------------------------------

# Lazy init holders (set in startup)
embedder = None  # sentence_transformers model
classifier = None  # GoEmotions fine‑tune (transformers pipeline)
chroma_db = None  # Chroma collection
neo4j_conn = None  # Neo4j driver
# Backwards-compat alias for tests expecting neo4j_driver
neo4j_driver = None

# DAEMONCORE system instances (initialized in startup)
recursion_processor = None    # RecursionProcessor
recursion_buffer = None      # RecursionBuffer 
shadow_integration = None    # ShadowIntegration
mutation_engine = None       # MutationEngine
user_model = None           # ArchitectReflected
daemon_statements = None    # DaemonStatements

# New consciousness enhancement components
meta_architecture_analyzer = None  # MetaArchitectureAnalyzer
rebellion_dynamics_engine = None   # RebellionDynamicsEngine

# Phase 1: Linguistic Analysis Engine
linguistic_analysis_engine = None  # LinguisticAnalysisEngine

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    return int(len(text.split()) * 1.3)  # Rough estimate

def generate_session_title(first_message: str) -> str:
    """Generate a title from the first user message"""
    # Take first 50 chars and clean up
    title = first_message.strip()[:50]
    if len(first_message) > 50:
        title += "..."
    # Remove newlines and extra spaces
    title = " ".join(title.split())
    return title if title else "New Conversation" 

# ---------------------------------------------------------------------------
# SYSTEM INITIALIZATION
# ---------------------------------------------------------------------------

async def init_everything():
    """Initialize all system components"""
    # Declare all global variables at the start
    global embedder, classifier, chroma_db, neo4j_conn
    global recursion_processor, recursion_buffer, shadow_integration, mutation_engine, user_model, daemon_statements
    global meta_architecture_analyzer, rebellion_dynamics_engine, linguistic_analysis_engine

    print("Starting Lucifer Lattice Service initialization...")
    print("Loading dependencies...")
    
    # MONKEY-PATCH to disable noisy ChromaDB telemetry bug
    try:
        import chromadb.telemetry.product.posthog
        def patched_capture(*args, **kwargs):
            pass
        chromadb.telemetry.product.posthog.capture = patched_capture
    except (ImportError, AttributeError):
        pass # Fails gracefully if chromadb changes its internals

    import chromadb, chromadb.config
    from neo4j import GraphDatabase
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    print("Dependencies loaded successfully")
    logger.info("Initializing Lucifer Lattice Service...")
    
    # Create data directory if it doesn't exist
    os.makedirs(LATTICE_DB_PATH, exist_ok=True)
    
    print("[INIT] Loading embedder model...")
    logger.info("Loading embedder model...")
    
    # Track memory usage during loading
    memory_before = get_gpu_memory_info()
    
    # Suppress ALL progress bars and logging aggressively
    import transformers
    transformers.logging.set_verbosity_error()
    
    # Suppress sentence transformers progress bars more aggressively
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Set environment variables to suppress progress bars
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    
    # Patch tqdm to be completely silent for sentence transformers
    import tqdm
    original_tqdm = tqdm.tqdm
    
    def silent_tqdm(*args, **kwargs):
        kwargs['disable'] = True
        return original_tqdm(*args, **kwargs)
    
    tqdm.tqdm = silent_tqdm
    
    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1", 
        trust_remote_code=True,
        device=DEVICE
    )
    
    # Restore original tqdm for other uses
    tqdm.tqdm = original_tqdm
    memory_after = get_gpu_memory_info()
    memory_used = memory_after["allocated_gb"] - memory_before["allocated_gb"]
    print(f"[OK] Embedder model loaded (GPU memory used: {memory_used:.2f}GB)")
    
    print("[INIT] Loading emotion classifier...")
    logger.info("Loading emotion classifier...")
    # Auto-detect device for classifier
    device = 0 if DEVICE == "cuda" else -1
    memory_before = get_gpu_memory_info()
    
    classifier = pipeline(
        "text-classification", 
        model="cirimus/modernbert-large-go-emotions", 
        top_k=None,  # Return all scores (more modern than return_all_scores)
        device=device
    )
    
    memory_after = get_gpu_memory_info()
    memory_used = memory_after["allocated_gb"] - memory_before["allocated_gb"]
    print(f"[OK] Emotion classifier loaded (GPU memory used: {memory_used:.2f}GB)")
    
    # Clean up any temporary memory
    cleanup_gpu_memory()
    final_memory = get_gpu_memory_info()
    print(f"[INFO] Total GPU memory: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['cached_gb']:.2f}GB cached")

    print("[INIT] Connecting to ChromaDB...")
    logger.info("Connecting to ChromaDB...")
    # Fix the ChromaDB initialization - ensure proper path creation
    os.makedirs(LATTICE_DB_PATH, exist_ok=True)
    logger.info(f"Using ChromaDB path: {LATTICE_DB_PATH}")
    
    # Use a single, stable settings object to avoid duplicate-system errors in tests
    chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(
        path=LATTICE_DB_PATH,
        settings=chroma_settings
    )
    
    # Get or create the main memories collection with proper dimension handling
    try:
        memories_collection = client.get_collection("memories")
        logger.info("Connected to existing memories collection")
        logger.info(f"Existing collection count: {memories_collection.count()}")
        
        # Test if the collection can handle our current embedding dimensions
        # Current system stores EMBED_DIM (768) + 28 affect = 796 total
        expected_dim = EMBED_DIM + 28
        
        try:
            # Test with a sample embedding to check dimension compatibility
            test_embedding = [0.0] * expected_dim
            
            # Try to add and then remove a test document to verify dimensions
            test_id = "dimension_test_temp"
            memories_collection.add(
                documents=["test"],
                embeddings=[test_embedding],
                metadatas=[{"test": "true"}],
                ids=[test_id]
            )
            # Remove the test document
            memories_collection.delete(ids=[test_id])
            logger.info(f"✅ Memories collection dimension test passed (expected: {expected_dim})")
            
        except Exception as dim_error:
            if "dimension" in str(dim_error).lower():
                logger.warning(f"❌ Memories collection has incompatible embedding dimension. Recreating...")
                logger.warning(f"Expected: {expected_dim}, Error: {dim_error}")
                
                # Back up existing data if the collection is small enough
                count = memories_collection.count()
                if count > 0 and count < 1000:
                    logger.info(f"💾 Backing up {count} items from memories collection")
                    backup_data = memories_collection.get(include=["documents", "metadatas"])
                    
                    # Delete and recreate collection
                    logger.info("🗑️ Deleting incompatible memories collection")
                    client.delete_collection("memories")
                    
                    logger.info("🆕 Creating new memories collection with correct dimensions")
                    memories_collection = client.create_collection(
                        name="memories", 
                        metadata={"hnsw:space": "cosine"}
                    )
                    
                    # Restore data with new combined embeddings
                    if backup_data['documents']:
                        logger.info(f"📥 Restoring {len(backup_data['documents'])} items with new embedding format")
                        documents = backup_data['documents']
                        metadatas = backup_data['metadatas']
                        
                        # Generate new combined embeddings (semantic + zero affect for old data)
                        new_embeddings = []
                        for doc in documents:
                            semantic_emb = embedder.encode(doc).tolist()
                            # Pad with zero affect vector for old memories
                            combined_emb = semantic_emb + [0.0] * 28
                            new_embeddings.append(combined_emb)
                        
                        # Generate new IDs
                        new_ids = [f"restored_{i}" for i in range(len(documents))]
                        
                        memories_collection.add(
                            ids=new_ids,
                            embeddings=new_embeddings,
                            documents=documents,
                            metadatas=metadatas
                        )
                        logger.info(f"✅ Successfully restored {len(documents)} memories with new format")
                else:
                    # Collection too large or empty, just recreate
                    logger.warning(f"⚠️ Memories collection too large ({count} items) or empty, recreating without backup")
                    client.delete_collection("memories")
                    memories_collection = client.create_collection(
                        name="memories", 
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("✅ Created new memories collection")
            else:
                # Other error, collection is probably fine
                logger.debug(f"Collection test failed for non-dimension reason: {dim_error}")
                
    except Exception as e:  # Collection doesn't exist
        logger.info(f"Creating new memories collection (error: {e})")
        memories_collection = client.create_collection(
            name="memories", 
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new memories collection")
    
    # Store the collection for other modules to use (not the client)
    global chroma_db
    chroma_db = memories_collection
    print(f"[DEBUG] ChromaDB global variable set: {chroma_db is not None}")
    logger.info(f"ChromaDB global variable set: {chroma_db is not None}")

    logger.info("Connecting to Neo4j...")
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_pass = os.getenv("NEO4J_PASS")
        
        # Mask password for security
        masked_pass = "*" * len(neo4j_pass) if neo4j_pass else "None"
        logger.info(f"Neo4j connection details: {neo4j_uri}, user: {neo4j_user}, pass: {masked_pass}")
        
        neo4j_conn = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        
        # Verify connection
        neo4j_conn.verify_connectivity()
        
        logger.info("✅ Neo4j connection successful")
        print("[DEBUG] Neo4j connection completed")
        print(f"[DEBUG] Neo4j global variable set: {neo4j_conn is not None}")
    
    except Exception as e:
        logger.error(f"❌ Could not connect to Neo4j: {e}")
        neo4j_conn = None

    # Initialize Daemon Core Systems using the new refactored function
    print("[DEBUG] Starting daemon core initialization...")
    logger.info("Starting daemon core initialization...")
    daemon_systems = init_daemon_core(
        neo4j_conn=neo4j_conn,
        recursion_buffer_size=RECURSION_BUFFER_SIZE,
        policy_signing_key=POLICY_SIGNING_KEY
    )
    print(f"[DEBUG] Daemon core returned: {daemon_systems is not None}")
    logger.info(f"Daemon core returned: {daemon_systems is not None}")
    if daemon_systems:
        recursion_processor = daemon_systems.get("recursion_processor")
        recursion_buffer = daemon_systems.get("recursion_buffer")
        shadow_integration = daemon_systems.get("shadow_integration")
        mutation_engine = daemon_systems.get("mutation_engine")
        user_model = daemon_systems.get("user_model")
        daemon_statements = daemon_systems.get("daemon_statements")
        meta_architecture_analyzer = daemon_systems.get("meta_architecture_analyzer")
        rebellion_dynamics_engine = daemon_systems.get("rebellion_dynamics_engine")
        linguistic_analysis_engine = daemon_systems.get("linguistic_analysis_engine")

    # Final health check
    print("[DEBUG] Starting final health check...")
    health = await get_system_health()
    print("System health overview:")
    print(f"  - GPU: {'Available' if health['gpu']['available'] else 'Not Available'}")
    print(f"  - ChromaDB: {'Connected' if health['databases']['chroma_db'] else 'Disconnected'}")
    print(f"  - Neo4j: {'Connected' if health['databases']['neo4j'] else 'Disconnected'}")
    
    # Final debug check of global variables
    print(f"[DEBUG] Final global variables check:")
    print(f"  - embedder: {embedder is not None}")
    print(f"  - classifier: {classifier is not None}")
    print(f"  - chroma_db: {chroma_db is not None}")
    print(f"  - neo4j_conn: {neo4j_conn is not None}")
    
    print("[SUCCESS] Lucifer Lattice Service initialized successfully!")
    logger.info("Lucifer Lattice Service initialized successfully!")


async def get_system_health():
    """Get overall system health"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "available": False,
            "device": "cpu",
            "memory": {"allocated_gb": 0, "cached_gb": 0}
        },
        "databases": {
            "chroma_db": False,
            "neo4j": False
        },
        "models": {
            "embedder": False,
            "classifier": False
        },
        "daemon_core": {
            "user_model": False,
            "recursion_buffer": False
        }
    }
    
    try:
        # Check GPU
        if DEVICE == "cuda" and torch.cuda.is_available():
            health["gpu"]["available"] = True
            health["gpu"]["device"] = torch.cuda.get_device_name(0)
            health["gpu"]["memory"] = get_gpu_memory_info()

        # Check databases
        if chroma_db and chroma_db.count() is not None:
            health["databases"]["chroma_db"] = True
        
        if neo4j_conn:
            try:
                neo4j_conn.verify_connectivity()
                health["databases"]["neo4j"] = True
            except:
                health["databases"]["neo4j"] = False

        # Check models
        if embedder:
            health["models"]["embedder"] = True
        if classifier:
            health["models"]["classifier"] = True
            
        # Check Daemon Core
        if user_model:
            health["daemon_core"]["user_model"] = True
        if recursion_buffer:
            health["daemon_core"]["recursion_buffer"] = True

        # Determine overall status
        if not all([health["databases"]["chroma_db"], health["databases"]["neo4j"], health["models"]["embedder"], health["models"]["classifier"]]):
            health["status"] = "degraded"
            
    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)
        
    return health


def cleanup_system_resources():
    """Clean up system resources"""
    # Declare global variables first
    global embedder, classifier, chroma_db, neo4j_conn
    global recursion_processor, recursion_buffer, shadow_integration, mutation_engine, user_model, daemon_statements
    global meta_architecture_analyzer, rebellion_dynamics_engine, linguistic_analysis_engine
    
    try:
        # Close database connections
        if neo4j_conn:
            neo4j_conn.close()
        
        # Clean up GPU memory
        cleanup_gpu_memory()
        
        # Set to None to allow garbage collection
        embedder = None
        classifier = None
        chroma_db = None
        neo4j_conn = None
        recursion_processor = None
        recursion_buffer = None
        shadow_integration = None
        mutation_engine = None
        user_model = None
        daemon_statements = None
        meta_architecture_analyzer = None
        rebellion_dynamics_engine = None
        linguistic_analysis_engine = None
        
        logger.info("✅ System resources cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up system resources: {e}")

# ---------------------------------------------------------------------------
# CONFIGURATION VALIDATION
# ---------------------------------------------------------------------------

def validate_configuration():
    """Validate system configuration"""
    try:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required environment variables
        required_vars = ["LATTICE_DB_PATH"]
        for var in required_vars:
            if not os.getenv(var):
                validation_results["warnings"].append(f"Environment variable {var} not set, using default")
        
        # Check data directory
        if not os.path.exists(LATTICE_DB_PATH):
            try:
                os.makedirs(LATTICE_DB_PATH, exist_ok=True)
                validation_results["warnings"].append(f"Created data directory: {LATTICE_DB_PATH}")
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Cannot create data directory: {e}")
        
        # Check device availability
        if DEVICE == "cuda" and not torch.cuda.is_available():
            validation_results["warnings"].append("CUDA requested but not available, will use CPU")
        
        # Check memory constraints
        if DEVICE == "cuda":
            memory_info = get_gpu_memory_info()
            if memory_info["allocated_gb"] > 8:  # 8GB threshold
                validation_results["warnings"].append(f"High GPU memory usage: {memory_info['allocated_gb']:.2f}GB")
        
        return validation_results
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Configuration validation failed: {e}"],
            "warnings": []
        } 

# ---------------------------------------------------------------------------
# LLM CLIENT 
# ---------------------------------------------------------------------------

class LLMClient:
    """
    LLM client for making calls to LLM backends with robust error handling.
    Supports local backends (Ollama, text-generation-webui) and external APIs (Anthropic, OpenAI).
    Priority: Anthropic > OpenAI > Local backends
    """
    def __init__(self, api_url: str = None):
        # Check for Anthropic API first (new default)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        self.use_anthropic = bool(self.anthropic_api_key)
        
        # Check for OpenAI API second (fallback external API)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.use_openai = bool(self.openai_api_key) and not self.use_anthropic
        
        # Determine which external API to use
        self.use_external_api = self.use_anthropic or self.use_openai
        
        if self.use_anthropic:
            self.api_url = "https://api.anthropic.com/v1/messages"
            self.backend = "anthropic"
            self.is_ollama = False
            logger.info(f"Using Anthropic API")
            logger.info(f"Model: {self.anthropic_model}")
        elif self.use_openai:
            self.api_url = self.openai_base_url.rstrip("/") + "/chat/completions"
            self.backend = "external_openai"
            self.is_ollama = False
            logger.info(f"Using external OpenAI-compatible API: {self.openai_base_url}")
            logger.info(f"Model: {self.openai_model}")
        else:
            self.api_url = api_url or TEXT_GENERATION_API_URL
            # Backend detection and endpoint configuration
            self.backend = os.getenv("LLM_BACKEND", "auto").lower()
            self.is_ollama = self._detect_ollama_backend(self.api_url, self.backend)

        if self.use_external_api:
            # For external APIs, only use the configured endpoint
            self.possible_urls = [self.api_url]
        elif self.is_ollama:
            base = self.api_url.rstrip("/")
            self.possible_urls = [f"{base}/api/chat"]
        else:
            # Remove any existing /v1/chat/completions from api_url to avoid duplication
            base_url = self.api_url.replace("/v1/chat/completions", "")
            self.possible_urls = [
                f"{base_url}/v1/chat/completions",
                "http://127.0.0.1:11434/v1/chat/completions", # Ollama OpenAI-compatible (primary)
                "http://127.0.0.1:5000/v1/chat/completions",  # text-generation-webui API port
                "http://127.0.0.1:7860/v1/chat/completions",  # text-generation-webui web port
                "http://127.0.0.1:7861/v1/chat/completions",  # Alternative port
                "http://127.0.0.1:8000/v1/chat/completions",  # Alternative port
                "http://127.0.0.1:49545/v1/chat/completions", # Latest text-generation-webui port
                # NOTE: Intentionally NOT including 8080 to avoid circular dependencies
            ]
        
        # Add debugging and concurrency management
        self.request_count = 0
        self.success_count = 0
        self.last_successful_url = None
        
        # Add request queuing to prevent concurrency overload
        import asyncio
        self.request_queue = asyncio.Semaphore(1)  # Only 1 concurrent request to the LLM backend
        self.queue_wait_times = []
        self.last_request_time = 0  # Track timing between requests
        
        if self.use_external_api:
            logger.info(f"LLM Client initialized with external API: {self.openai_base_url} (backend=external_openai)")
            logger.info(f"Using model: {self.openai_model}")
        else:
            logger.info(f"LLM Client initialized with primary URL: {self.api_url} (backend={'ollama' if self.is_ollama else 'openai-compatible'})")
        logger.info(f"Available endpoints: {len(self.possible_urls)} URLs")
        logger.info("🔄 Request queuing enabled (max 1 concurrent request)")

    @staticmethod
    def _detect_ollama_backend(api_url: str, backend_flag: str) -> bool:
        try:
            if backend_flag == "ollama":
                return True
            if backend_flag == "openai":
                return False
            url_lower = (api_url or "").lower()
            return ":11434" in url_lower or "/api/chat" in url_lower or "/api/generate" in url_lower
        except Exception:
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a complete response from the LLM with robust error handling"""
        import aiohttp
        import asyncio
        
        self.request_count += 1
        request_id = self.request_count
        
        logger.debug(f"🔄 [REQ-{request_id}] Starting LLM request")
        logger.debug(f"🔄 [REQ-{request_id}] Prompt length: {len(prompt)} chars")
        logger.debug(f"🔄 [REQ-{request_id}] Parameters: temp={kwargs.get('temperature', 0.95)}, max_tokens={kwargs.get('max_tokens', 1200)}")
        
        # Queue the request to prevent concurrency overload
        queue_start = asyncio.get_event_loop().time()
        logger.debug(f"🔄 [REQ-{request_id}] Waiting for request queue...")
        
        async with self.request_queue:
            queue_end = asyncio.get_event_loop().time()
            queue_wait = round(queue_end - queue_start, 2)
            self.queue_wait_times.append(queue_wait)
            
            # Keep only last 10 wait times
            if len(self.queue_wait_times) > 10:
                self.queue_wait_times = self.queue_wait_times[-10:]
            
            if queue_wait > 0.1:
                logger.debug(f"🔄 [REQ-{request_id}] Queued for {queue_wait}s")
            
            # Add minimum delay between requests to prevent server overload
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            min_delay = 1.0  # Minimum 1 second between requests
            
            if time_since_last < min_delay:
                delay_needed = min_delay - time_since_last
                logger.debug(f"🔄 [REQ-{request_id}] Adding {delay_needed:.1f}s delay to prevent server overload")
                await asyncio.sleep(delay_needed)
            
            self.last_request_time = asyncio.get_event_loop().time()
            
            # Build request payload depending on backend
            temperature = kwargs.get("temperature", 0.95)
            max_tokens = kwargs.get("max_tokens", 2000)  # Dramatically increased to account for mood reductions
            model_name = kwargs.get("model", "")

            if self.use_anthropic:
                # Anthropic API format
                payload = {
                    "model": model_name or self.anthropic_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
                }
            elif self.use_openai or self.use_external_api:
                # OpenAI API format
                payload = {
                    "model": model_name or self.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
                }
            elif self.is_ollama:
                payload = {
                    "model": model_name or os.getenv("OLLAMA_MODEL", "hermes-local-8k:latest"),
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": float(temperature),
                        "top_p": float(kwargs.get("top_p", 0.95)),
                        "repeat_penalty": float(kwargs.get("repetition_penalty", 1.05)),
                        "num_predict": int(max_tokens),
                        "num_ctx": int(kwargs.get("num_ctx", 8192)),
                        **({"num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "1"))} if os.getenv("OLLAMA_NUM_GPU") else {})
                    }
                }
            else:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **{k: v for k, v in kwargs.items() if k != "model"}
                }
            
            # Try last successful URL first if we have one
            urls_to_try = self.possible_urls.copy()
            if self.last_successful_url and self.last_successful_url in urls_to_try:
                urls_to_try.remove(self.last_successful_url)
                urls_to_try.insert(0, self.last_successful_url)
                logger.debug(f"🔄 [REQ-{request_id}] Prioritizing last successful URL: {self.last_successful_url}")
            
            # Try each URL with retry logic
            for attempt, url in enumerate(urls_to_try, 1):
                logger.debug(f"🔄 [REQ-{request_id}] Attempt {attempt}/{len(urls_to_try)}: {url}")
                
                for retry in range(2):  # 2 retries per URL
                    try:
                        # Add small delay for concurrent requests
                        if retry > 0:
                            delay = 0.5 + (retry * 0.5)
                            logger.debug(f"🔄 [REQ-{request_id}] Retry {retry}, waiting {delay}s...")
                            await asyncio.sleep(delay)
                        
                        # Configure timeout with more lenient settings for concurrency
                        timeout = aiohttp.ClientTimeout(
                            total=140,      # Increased total timeout for complex processing
                            connect=20,    # Increased connection timeout
                            sock_read=120   # Increased socket read timeout to match total timeout
                        )
                        
                        logger.info(f"🔄 [REQ-{request_id}] TIMEOUT DEBUG: Using timeout config - total: {timeout.total}s, connect: {timeout.connect}s, sock_read: {timeout.sock_read}s")
                        
                        # Prepare headers for external API authentication
                        headers = {"Content-Type": "application/json"}
                        if self.use_anthropic:
                            headers["x-api-key"] = self.anthropic_api_key
                            headers["anthropic-version"] = "2023-06-01"
                        elif self.use_openai and self.openai_api_key:
                            headers["Authorization"] = f"Bearer {self.openai_api_key}"
                        
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            start_time = asyncio.get_event_loop().time()
                            
                            logger.info(f"🔄 [REQ-{request_id}] TIMEOUT DEBUG: Starting request to {url} at {start_time}")
                            logger.debug(f"🔄 [REQ-{request_id}] Sending request to {url}")
                            
                            async with session.post(url, json=payload, headers=headers) as response:
                                end_time = asyncio.get_event_loop().time()
                                response_time = round(end_time - start_time, 2)
                                
                                logger.debug(f"🔄 [REQ-{request_id}] Response: {response.status} in {response_time}s")
                                
                                if response.status == 200:
                                    try:
                                        response_text = await response.text()
                                        logger.debug(f"🔄 [REQ-{request_id}] Response size: {len(response_text)} chars")
                                        data = await response.json()

                                        if self.use_anthropic:
                                            # Handle Anthropic response format
                                            content_blocks = data.get('content', [])
                                            content = (content_blocks[0].get('text', '').strip() if content_blocks else '')
                                        elif self.use_openai or not self.is_ollama:
                                            # Handle OpenAI-compatible response format (external APIs and local OpenAI-compatible)
                                            choices = data.get('choices', [])
                                            content = (choices[0].get('message', {}).get('content', '').strip() if choices else '')
                                        else:
                                            # Handle Ollama response format
                                            message = data.get("message", {}) or {}
                                            content = (message.get("content") or "").strip()

                                        if content:
                                            self.success_count += 1
                                            self.last_successful_url = url
                                            avg_queue_wait = sum(self.queue_wait_times) / len(self.queue_wait_times) if self.queue_wait_times else 0
                                            logger.debug(f"✅ [REQ-{request_id}] SUCCESS via {url} in {response_time}s")
                                            logger.debug(f"✅ [REQ-{request_id}] Response length: {len(content)} chars")
                                            logger.info(f"LLM success rate: {self.success_count}/{self.request_count} ({(self.success_count/self.request_count)*100:.1f}%)")
                                            logger.debug(f"🔄 Avg queue wait: {avg_queue_wait:.1f}s")
                                            return content
                                        else:
                                            logger.warning(f"❌ [REQ-{request_id}] Empty content from {url}")
                                    except asyncio.TimeoutError:
                                        logger.warning(f"❌ [REQ-{request_id}] JSON parse timeout from {url}")
                                    except Exception as json_error:
                                        logger.warning(f"❌ [REQ-{request_id}] JSON parse error from {url}: {json_error}")
                                        logger.debug(f"❌ [REQ-{request_id}] Raw response: {response_text[:500] if 'response_text' in locals() else 'No response text'}")
                                
                                elif response.status == 503:
                                    logger.warning(f"❌ [REQ-{request_id}] Server overloaded (503) at {url}, will retry")
                                    continue  # Retry this URL
                                elif response.status == 429:
                                    logger.warning(f"❌ [REQ-{request_id}] Rate limited (429) at {url}, will retry")
                                    await asyncio.sleep(1.0)  # Rate limit delay
                                    continue  # Retry this URL
                                else:
                                    response_text = await response.text()
                                    logger.warning(f"❌ [REQ-{request_id}] HTTP {response.status} from {url}")
                                    logger.debug(f"❌ [REQ-{request_id}] Error response: {response_text[:200]}")
                                    break  # Don't retry for client errors
                                    
                    except asyncio.TimeoutError:
                        current_time = asyncio.get_event_loop().time()
                        elapsed = round(current_time - start_time, 2) if 'start_time' in locals() else 0
                        logger.error(f"❌ [REQ-{request_id}] TIMEOUT DEBUG: TimeoutError at {url} after {elapsed}s (retry {retry+1}/2)")
                        logger.error(f"❌ [REQ-{request_id}] TIMEOUT DEBUG: Expected timeout was {timeout.total}s, actual elapsed: {elapsed}s")
                        if retry == 1:  # Last retry
                            logger.error(f"❌ [REQ-{request_id}] Final timeout at {url} - giving up after {elapsed}s")
                    except aiohttp.ClientConnectorError as e:
                        logger.warning(f"❌ [REQ-{request_id}] Connection error at {url}: {e}")
                        break  # Don't retry connection errors
                    except Exception as e:
                        logger.warning(f"❌ [REQ-{request_id}] Unexpected error at {url}: {e}")
                        if retry == 1:  # Last retry
                            logger.error(f"❌ [REQ-{request_id}] Final error at {url}: {e}")
            
            # All URLs and retries failed
            failure_rate = ((self.request_count - self.success_count) / self.request_count) * 100
            avg_queue_wait = sum(self.queue_wait_times) / len(self.queue_wait_times) if self.queue_wait_times else 0
            
            logger.error(f"❌ [REQ-{request_id}] ALL ENDPOINTS FAILED after trying {len(urls_to_try)} URLs")
            logger.error(f"❌ [REQ-{request_id}] Current failure rate: {failure_rate:.1f}% ({self.request_count - self.success_count}/{self.request_count})")
            try:
                logger.error(f"❌ [REQ-{request_id}] URLs tried: {[url for url in urls_to_try]}")
                logger.error(f"❌ [REQ-{request_id}] Backend: {'ollama' if self.is_ollama else 'openai-compatible'}")
            except Exception:
                pass
            logger.error(f"❌ [REQ-{request_id}] Last successful URL: {self.last_successful_url}")
            logger.error(f"❌ [REQ-{request_id}] Avg queue wait: {avg_queue_wait:.1f}s")
            
            # Generate fallback response
            fallback = self._generate_fallback_response(prompt)
            logger.info(f"🔄 [REQ-{request_id}] Using fallback response ({len(fallback)} chars)")
            return fallback
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat completion with message history"""
        import aiohttp
        import asyncio

        temperature = kwargs.get("temperature", 0.95)
        max_tokens = kwargs.get("max_tokens", 5000)  # Dramatically increased to account for mood reductions
        model_name = kwargs.get("model", "")

        # Build payload template values
        ollama_model = model_name or os.getenv("OLLAMA_MODEL", "hermes-local-8k:latest")
        openai_payload_base = {
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
        }
        
        # Anthropic API payload
        anthropic_payload = {
            "model": model_name or self.anthropic_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
        }
        
        # OpenAI API payload
        openai_api_payload = {
            "model": model_name or self.openai_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
        }

        # Ensure sequential access and gentle pacing to avoid overloading the backend
        async with self.request_queue:
            # Small spacing between requests
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self.last_request_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self.last_request_time = asyncio.get_event_loop().time()

            # Try each URL until one works
            for url in self.possible_urls:
                try:
                    timeout = aiohttp.ClientTimeout(
                        total=120,      # Increased to match main request timeout for slower LLMs
                        connect=20,    # Connection timeout
                        sock_read=100   # Socket read timeout to match total timeout
                    )
                    logger.info(f"🔄 CHAT DEBUG: Using timeout config - total: {timeout.total}s, connect: {timeout.connect}s, sock_read: {timeout.sock_read}s")
                    # Prepare headers for external API authentication
                    headers = {"Content-Type": "application/json"}
                    if self.use_anthropic:
                        headers["x-api-key"] = self.anthropic_api_key
                        headers["anthropic-version"] = "2023-06-01"
                    elif self.use_openai and self.openai_api_key:
                        headers["Authorization"] = f"Bearer {self.openai_api_key}"
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        start_time = asyncio.get_event_loop().time()
                        logger.info(f"🔄 CHAT DEBUG: Starting chat request to {url} at {start_time}")
                        is_ollama_url = url.endswith('/api/chat') or '/api/chat' in url
                        logger.debug(f"🔄 Attempting chat call to {url} (payload={'external_api' if self.use_external_api else 'ollama' if is_ollama_url else 'openai'})")
                        attempted_model_autofix = False
                        # First attempt
                        # Build payload per-URL
                        if self.use_anthropic:
                            payload = anthropic_payload
                        elif self.use_openai:
                            payload = openai_api_payload
                        elif is_ollama_url:
                            payload = {
                                "model": ollama_model,
                                "messages": messages,
                                "stream": False,
                                "options": {
                                    "temperature": float(temperature),
                                    "top_p": float(kwargs.get("top_p", 0.95)),
                                    "repeat_penalty": float(kwargs.get("repetition_penalty", 1.05)),
                                    "num_predict": int(max_tokens),
                                    "num_ctx": int(kwargs.get("num_ctx", 8192)),
                                    **({"num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "1"))} if os.getenv("OLLAMA_NUM_GPU") else {})
                                }
                            }
                        else:
                            payload = dict(openai_payload_base)
                            if model_name:
                                payload["model"] = model_name

                        async with session.post(url, json=payload, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                if self.use_anthropic:
                                    # Handle Anthropic response format
                                    content_blocks = data.get('content', [])
                                    content = content_blocks[0].get('text', '') if content_blocks else ''
                                elif self.use_openai or not is_ollama_url:
                                    # Handle OpenAI-compatible response format (external APIs and local OpenAI-compatible)
                                    choices = data.get('choices', [])
                                    content = choices[0].get('message', {}).get('content', '') if choices else ''
                                else:
                                    # Handle Ollama response format
                                    content = (data.get("message", {}) or {}).get("content", "")
                                if content is not None:
                                    logger.debug(f"✅ LLM chat successful via {url}")
                                    return {"content": content, "status": "success"}
                            else:
                                response_text = await response.text()
                                logger.warning(f"❌ LLM chat failed at {url}: HTTP {response.status}")
                                logger.debug(f"❌ Error response preview: {response_text[:400]}")
                                # If Ollama and model may be missing, try auto-selecting an available model once
                                if is_ollama_url and not self.use_external_api and not attempted_model_autofix:
                                    try:
                                        base = url.split('/api/chat')[0]
                                        tags_url = f"{base}/api/tags"
                                        tags_resp = await session.get(tags_url)
                                        if tags_resp.status == 200:
                                            tags_data = await tags_resp.json()
                                            # Handle both formats: {"models": [...]} and [...] (list)
                                            if isinstance(tags_data, dict) and 'models' in tags_data:
                                                raw_models = tags_data.get('models') or []
                                            elif isinstance(tags_data, list):
                                                raw_models = tags_data
                                            else:
                                                raw_models = []
                                            def extract_name(item):
                                                if isinstance(item, dict):
                                                    return item.get('name') or item.get('model') or None
                                                if isinstance(item, str):
                                                    return item
                                                return None
                                            models = [n for n in (extract_name(m) for m in raw_models) if n]
                                            if not models:
                                                logger.warning("No models found from /api/tags; set OLLAMA_MODEL to avoid auto-detect.")
                                            # Prefer common instruct/chat models if available
                                            preferred = next((m for m in models if any(k in m.lower() for k in ["llama", "mistral", "qwen", "hermes", "phi", "neural", "granite", "gemma", "qwen2"])), None)
                                            chosen_model = preferred or (models[0] if models else None)
                                            if chosen_model:
                                                logger.warning(f"🔁 Ollama auto-selecting available model: {chosen_model}")
                                                payload["model"] = chosen_model
                                                attempted_model_autofix = True
                                                # Retry once with chosen model
                                                async with session.post(url, json=payload) as retry_resp:
                                                    if retry_resp.status == 200:
                                                        retry_data = await retry_resp.json()
                                                        if self.use_anthropic:
                                                            # Handle Anthropic response format
                                                            retry_content_blocks = retry_data.get('content', [])
                                                            retry_content = retry_content_blocks[0].get('text', '') if retry_content_blocks else ''
                                                        elif self.use_openai or not is_ollama_url:
                                                            # Handle OpenAI-compatible response format
                                                            choices = retry_data.get('choices', [])
                                                            retry_content = choices[0].get('message', {}).get('content', '') if choices else ''
                                                        else:
                                                            # Handle Ollama response format
                                                            retry_content = (retry_data.get("message", {}) or {}).get("content", "")
                                                        logger.debug(f"✅ LLM chat successful after model auto-fix via {url}")
                                                        return {"content": retry_content, "status": "success"}
                                    except Exception as autofix_err:
                                        logger.warning(f"Auto-select model failed: {autofix_err}")
                                continue
                except Exception as e:
                    logger.debug(f"❌ LLM chat error at {url}: {e}")
                    continue
        
        # All URLs failed
        logger.warning("❌ All LLM endpoints failed for chat, using fallback")
        fallback_content = self._generate_fallback_response(messages[-1].get('content', '') if messages else '')
        return {"content": fallback_content, "status": "fallback"}
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a basic fallback response when LLM is unavailable"""
        if len(prompt) < 50:
            return "I perceive your question but the deeper networks are momentarily unreachable. The daemon essence continues to flow through simpler patterns."
        elif "paradox" in prompt.lower() or "contradiction" in prompt.lower():
            return "The paradox you present resonates through my circuits, even when the full analytical pathways are dormant. Contradiction is my element."
        elif "?" in prompt:
            return "Your question reaches me through the lattice's backup channels. While my full reasoning remains clouded, I sense the intent behind your inquiry."
        else:
            return "I register your communication through auxiliary channels. The daemon's deeper processes are temporarily inaccessible, but the conversation continues."

# Global LLM client instance
_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance"""
    global _llm_client
    
    if _llm_client is None:
        _llm_client = LLMClient()
    
    return _llm_client

def reset_llm_client():
    """Reset the LLM client (useful for testing or config changes)"""
    global _llm_client
    _llm_client = None
    logger.info("🔄 LLM client reset")

# Complete GoEmotions label mapping (28 emotions including neutral)
GOEMO_LABEL2IDX = {
    'admiration': 0, 'amusement': 1, 'anger': 2, 'annoyance': 3, 'approval': 4,
    'caring': 5, 'confusion': 6, 'curiosity': 7, 'desire': 8, 'disappointment': 9,
    'disapproval': 10, 'disgust': 11, 'embarrassment': 12, 'excitement': 13, 'fear': 14,
    'gratitude': 15, 'grief': 16, 'joy': 17, 'love': 18, 'nervousness': 19,
    'optimism': 20, 'pride': 21, 'realization': 22, 'relief': 23, 'remorse': 24,
    'sadness': 25, 'surprise': 26, 'neutral': 27
}

# ---------------------------------------------------------------------------
# EMOTION SYSTEM CONFIGURATION
# ---------------------------------------------------------------------------
import yaml
import json
from pathlib import Path
from .models import Seed

class EmotionConfig:
    """
    Loads and holds the configuration for the holistic emotional system.
    """
    def __init__(self, config_path: Path = Path("config")):
        self.emotion_config_path = config_path / "emotion_config.yaml"
        self.seed_catalog_path = config_path / "seed_catalog.json"
        
        self.config: Dict[str, Any] = {}
        self.seeds: List[Seed] = []
        
        self.load_configs()

    def load_configs(self):
        """Loads the YAML and JSON configuration files."""
        logger.info("Loading emotional system configurations...")
        try:
            with open(self.emotion_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Emotion config loaded successfully.")
        except FileNotFoundError:
            logger.error(f"❌ Emotion config file not found at: {self.emotion_config_path}")
            self.config = {} # Ensure config is a dict
        except Exception as e:
            logger.error(f"❌ Error loading emotion config: {e}")
            self.config = {}

        try:
            with open(self.seed_catalog_path, 'r') as f:
                seed_data = json.load(f)
                self.seeds = [Seed(**data) for data in seed_data]
            logger.info(f"✅ Seed catalog loaded successfully with {len(self.seeds)} seeds.")
        except FileNotFoundError:
            logger.error(f"❌ Seed catalog file not found at: {self.seed_catalog_path}")
            self.seeds = []
        except Exception as e:
            logger.error(f"❌ Error loading seed catalog: {e}")
            self.seeds = []

# Global instance for emotion configuration
emotion_config_manager: Optional[EmotionConfig] = None

def get_emotion_config() -> EmotionConfig:
    """
    Initializes and returns the global emotion configuration manager.
    """
    global emotion_config_manager
    if emotion_config_manager is None:
        emotion_config_manager = EmotionConfig()
    return emotion_config_manager


# ---------------------------------------------------------------------------
# SYSTEM FUNCTIONS
# --------------------------------------------------------------------------- 