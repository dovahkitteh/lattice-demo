import logging
from datetime import datetime, timezone

from ..config import get_system_health

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HEALTH AND SYSTEM ENDPOINTS
# ---------------------------------------------------------------------------

async def health_check():
    """Health check endpoint for monitoring"""
    # Import globals at runtime to get current values
    from ..config import embedder, classifier, chroma_db, neo4j_conn
    
    status = {
        "status": "healthy",
        "service": "Lucifer Lattice Service",
        "components": {
            "embedder": embedder is not None,
            "classifier": classifier is not None,
            "chroma_db": chroma_db is not None,
            "neo4j_conn": neo4j_conn is not None
        }
    }
    return status

async def docs_redirect():
    """Redirect to API documentation"""
    return {"message": "Lucifer Lattice Service", "docs": "/docs", "health": "/health"}

async def get_gpu_status():
    """Get GPU status information"""
    try:
        health = get_system_health()
        return {
            "gpu_available": health["gpu"]["available"],
            "gpu_memory": health["gpu"]["memory"],
            "device": health["gpu"]["device"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {"error": str(e)}

async def cleanup_gpu():
    """Clean up GPU memory"""
    try:
        from ..config import cleanup_gpu_memory
        cleanup_gpu_memory()
        return {"status": "success", "message": "GPU memory cleaned"}
    except Exception as e:
        logger.error(f"Error cleaning GPU: {e}")
        return {"error": str(e)}