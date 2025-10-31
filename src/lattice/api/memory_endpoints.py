import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import HTTPException

from ..config import (
    neo4j_conn, GOEMO_LABEL2IDX
)
from ..memory import (
    get_storage_config, validate_unified_storage,
    store_emotional_memory_seed, get_recent_memories as get_recent_memories_core
)
from ..emotions import (
    classify_user_affect, get_emotional_influence
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEMORY ENDPOINTS
# ---------------------------------------------------------------------------

async def store_emotional_seed(seed_data: Dict[str, Any]):
    """
    Store an emotional memory seed with full symbolic preservation.
    
    Accepts rich emotional schema data and integrates it into the memory lattice
    while preserving poetic content and symbolic architecture.
    """
    try:
        # Validate required fields
        if "emotional_memory_seed" not in seed_data:
            raise HTTPException(status_code=400, detail="Missing 'emotional_memory_seed' field")
        
        # Extract the seed data
        seed = seed_data["emotional_memory_seed"]
        
        # Validate basic structure
        required_fields = ["metadata", "user_perspective", "ai_perspective"]
        for field in required_fields:
            if field not in seed:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Store the emotional memory seed
        node_id = await store_emotional_memory_seed(seed)
        
        return {
            "status": "success",
            "message": "Emotional memory seed stored successfully",
            "node_id": node_id,
            "title": seed.get("metadata", {}).get("title", "Untitled"),
            "category": seed.get("metadata", {}).get("category", "unknown"),
            "significance": seed.get("technical_mapping", {}).get("estimated_affect_magnitude", {}).get("total_significance", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing emotional memory seed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def upload_emotional_seed_file(seed_data: Dict[str, Any]):
    """
    Upload and store an emotional memory seed from a JSON file.
    
    This endpoint handles file uploads containing emotional memory seed JSON data
    and integrates them into the memory lattice.
    """
    try:
        # This endpoint receives the already-parsed JSON data from the frontend
        # The frontend handles the file reading and JSON parsing
        return await store_emotional_seed(seed_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading emotional memory seed file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def memory_stats():
    """Get statistics about stored memories"""
    try:
        # Import chroma_db at runtime to get current value
        from ..config import chroma_db
        
        if not chroma_db:
            return {"error": "ChromaDB not available", "count": 0}
        
        count = chroma_db.count()
        
        sample_data = {"documents": [], "metadatas": []}
        if count > 0:
            try:
                # Get all memories and sort by timestamp to find the most recent
                all_memories = chroma_db.get(include=["documents", "metadatas"])
                
                # Sort memories by timestamp in descending order
                sorted_memories = sorted(
                    zip(all_memories["ids"], all_memories["documents"], all_memories["metadatas"]),
                    key=lambda x: x[2].get("timestamp", "1970-01-01T00:00:00.000000+00:00"),
                    reverse=True
                )
                
                # Take the top 3 most recent memories for the sample
                recent_memories = sorted_memories[:3]
                
                docs = [mem[1] for mem in recent_memories]
                metas = [mem[2] for mem in recent_memories]
                
                sample_data = {
                    "documents": [doc[:100] + "..." if len(doc) > 100 else doc for doc in docs],
                    "metadatas": [
                        {
                            "origin": meta.get("origin", "unknown"),
                            "has_user_affect": "user_affect" in meta,
                            "has_self_affect": "self_affect" in meta,
                            "has_legacy_reflection": "reflection" in meta and bool(meta.get("reflection")),
                            "has_user_reflection": "user_reflection" in meta and bool(meta.get("user_reflection")),
                            "has_self_reflection": "self_reflection" in meta and bool(meta.get("self_reflection")),
                            "timestamp": meta.get("timestamp"), # For debugging
                            # New unified storage fields
                            "node_type": meta.get("node_type", "unknown"),
                            "lifecycle_state": meta.get("lifecycle_state", "unknown"),
                            "emotional_significance": meta.get("emotional_significance", 0.0),
                            "echo_count": meta.get("echo_count", 0)
                        } for meta in metas
                    ]
                }
            except Exception as e:
                logger.error(f"Error processing memory samples: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Get unified storage statistics
        storage_config = get_storage_config()
        
        return {
            "status": "success",
            "total_memories": count,
            "sample_memories": sample_data,
            "database_components": {
                "chroma_db": chroma_db is not None,
                "neo4j_conn": neo4j_conn is not None
            },
            "unified_storage": storage_config
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"error": str(e), "count": 0}

async def get_unified_storage_status():
    """Get detailed status of the unified storage system"""
    try:
        # Get configuration
        config = get_storage_config()
        
        # Get diagnostics - Note: This function might not exist, need to check
        # diagnostics = await get_unified_storage_diagnostics()
        
        return {
            "status": "success",
            "configuration": config,
            # "diagnostics": diagnostics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting unified storage status: {e}")
        return {"error": str(e)}

async def validate_unified_storage_endpoint():
    """Validate the unified storage system"""
    try:
        validation_results = await validate_unified_storage()
        return {
            "status": "success",
            "validation_results": validation_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error validating unified storage: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# EMOTION ENDPOINTS
# ---------------------------------------------------------------------------

async def analyze_emotions(request: dict):
    """Analyze emotions from text input"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        # Analyze emotions
        user_affect = await classify_user_affect(text)
        
        # Convert to emotion dictionary
        emotions = {}
        for i, score in enumerate(user_affect):
            emotion_name = [k for k, v in GOEMO_LABEL2IDX.items() if v == i][0]
            emotions[emotion_name] = score
        
        # Get emotional influence
        influence = await get_emotional_influence(user_affect)
        
        return {
            "emotions": emotions,
            "influence": influence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing emotions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_emotional_influence_for_text(request: dict):
    """Get emotional influence for given text"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
        
        # Analyze emotions
        user_affect = await classify_user_affect(text)
        
        # Get influence
        influence = await get_emotional_influence(user_affect)
        
        return {
            "text": text,
            "influence": influence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting emotional influence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_recent_memories(limit: int = 50):
    """Get recent memories from the database"""
    try:
        memories = await get_recent_memories_core(limit)
        return {
            "status": "success",
            "memories": memories,
            "count": len(memories),
            "limit": limit,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))