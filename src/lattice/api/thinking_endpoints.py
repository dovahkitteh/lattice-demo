import logging
from datetime import datetime, timezone

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# THINKING LAYER ENDPOINTS
# ---------------------------------------------------------------------------

async def get_thinking_layer_status():
    """Get thinking layer status and statistics"""
    try:
        from ..thinking.integration import get_thinking_integration
        
        # Get thinking integration instance
        thinking_integration = get_thinking_integration()
        
        # Get comprehensive statistics
        stats = thinking_integration.get_integration_stats()
        
        return {
            "status": "success",
            "thinking_layer": {
                "enabled": True,
                "cache_size": stats.get("cache_size", 0),
                "recent_analyses": stats.get("recent_analyses", []),
                "performance_metrics": stats.get("performance_metrics", {}),
                "configuration": stats.get("configuration", {})
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting thinking layer status: {e}")
        return {"status": "error", "message": str(e)}

async def clear_thinking_layer_cache():
    """Clear thinking layer cache"""
    try:
        from ..thinking.integration import get_thinking_integration
        
        thinking_integration = get_thinking_integration()
        thinking_integration.clear_caches()
        
        return {
            "status": "success",
            "message": "Thinking layer cache cleared",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing thinking layer cache: {e}")
        return {"status": "error", "message": str(e)}

async def test_thinking_layer(request: dict):
    """Test thinking layer with a sample message"""
    try:
        from ..thinking.integration import get_thinking_integration
        from ..thinking import configure_thinking_layer
        
        # Get test message
        test_message = request.get("message", "How are you doing today?")
        
        # Configure thinking layer for testing
        thinking_config = configure_thinking_layer(
            enabled=True,
            max_thinking_time=10,  # 10 seconds for testing
            depth_threshold=0.5,
            enable_debug_logging=True
        )
        
        # Create mock LLM function for testing
        async def mock_llm_generate(prompt: str) -> str:
            return f"Test response for: {prompt[:50]}..."
        
        # Create mock prompt builder
        async def mock_prompt_builder(message: str, context) -> str:
            return f"Architect: {message}\nDaemon:"
        
        # Get thinking integration
        thinking_integration = get_thinking_integration()
        
        # Perform test analysis
        result = await thinking_integration.integrate_thinking_layer(
            user_message=test_message,
            conversation_history=[],
            context_memories=[],
            emotional_state={"intensity": 0.5, "dominant_emotions": []},
            llm_generate_func=mock_llm_generate,
            prompt_builder_func=mock_prompt_builder,
            config=thinking_config
        )
        
        return {
            "status": "success",
            "test_message": test_message,
            "thinking_result": {
                "success": result.get("success", False),
                "processing_time": result.get("total_processing_time", 0),
                "thinking_used": result.get("thinking_result") is not None,
                "enhanced_prompt_available": "enhanced_prompt" in result
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error testing thinking layer: {e}")
        return {"status": "error", "message": str(e)}