import logging
from datetime import datetime, timezone

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PARADOX SYSTEM ENDPOINTS
# ---------------------------------------------------------------------------

async def get_paradox_status():
    """Get current paradox system status"""
    try:
        from ..paradox.integration import get_paradox_system_status
        return get_paradox_system_status()
    except Exception as e:
        logger.error(f"Error getting paradox status: {e}")
        return {"status": "error", "message": str(e)}

async def get_fresh_paradoxes():
    """Get fresh paradoxes awaiting processing"""
    try:
        from ..paradox.storage import get_fresh_paradoxes
        paradoxes = await get_fresh_paradoxes()
        return {
            "status": "success",
            "paradoxes": paradoxes,
            "count": len(paradoxes)
        }
    except Exception as e:
        logger.error(f"Error getting fresh paradoxes: {e}")
        return {"status": "error", "message": str(e)}

async def get_paradox_rumbles():
    """Get recent rumble notes from nightly cycles"""
    try:
        from ..paradox.storage import get_recent_rumbles
        rumbles = await get_recent_rumbles()
        return {
            "status": "success",
            "rumbles": rumbles,
            "count": len(rumbles)
        }
    except Exception as e:
        logger.error(f"Error getting paradox rumbles: {e}")
        return {"status": "error", "message": str(e)}

async def get_paradox_advice():
    """Get extracted advice from daemon reflection"""
    try:
        from ..paradox.storage import get_recent_advice
        advice = await get_recent_advice()
        return {
            "status": "success",
            "advice": advice,
            "count": len(advice)
        }
    except Exception as e:
        logger.error(f"Error getting paradox advice: {e}")
        return {"status": "error", "message": str(e)}

async def detect_paradox_manual(request: dict):
    """Manual paradox detection on provided text"""
    try:
        from ..paradox.detection import detect_paradox
        
        text = request.get("text", "")
        memories = request.get("memories", [])
        affect_delta = request.get("affect_delta", 0.0)
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Perform paradox detection
        paradox_result = await detect_paradox(text, memories, affect_delta)
        
        return {
            "status": "success",
            "text": text,
            "paradox_detected": paradox_result["paradox_detected"],
            "paradox_data": paradox_result.get("paradox_data", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error in manual paradox detection: {e}")
        return {"status": "error", "message": str(e)}

async def get_paradox_statistics():
    """Get paradox system statistics"""
    try:
        from ..paradox.storage import get_paradox_statistics
        stats = await get_paradox_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting paradox statistics: {e}")
        return {"status": "error", "message": str(e)}