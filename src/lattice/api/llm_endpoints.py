# src/lattice/api/llm_endpoints.py

"""
LLM integration endpoints for the Lucifer Lattice Service.
"""

import logging
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from fastapi.responses import PlainTextResponse

from ..config import get_llm_client, reset_llm_client
from ..model_manager import model_manager

logger = logging.getLogger(__name__)

class LLMTestRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 800

async def get_llm_status():
    """Get LLM client connection status"""
    try:
        client = get_llm_client()
        
        # Test connection with a simple prompt
        test_result = await client.generate_response("Hello", max_tokens=50)
        
        status = {
            "status": "connected",
            "api_url": client.api_url,
            "available_endpoints": client.possible_urls,
            "test_response_length": len(test_result) if test_result else 0,
            "connection_working": bool(test_result and len(test_result) > 10)
        }
        
        return status
        
    except Exception as e:
        logger.error(f"LLM status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "api_url": None,
            "connection_working": False
        }

async def test_llm_connection(request: LLMTestRequest):
    """Test LLM connection with a custom prompt"""
    try:
        client = get_llm_client()
        
        result = await client.generate_response(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "status": "success",
            "prompt": request.prompt,
            "response": result,
            "response_length": len(result) if result else 0
        }
        
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")

async def get_llm_endpoints():
    """Get available LLM endpoints and their status"""
    try:
        client = get_llm_client()
        
        endpoint_status = []
        for url in client.possible_urls:
            try:
                # Quick test of each endpoint
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    test_payload = {
                        "model": "test",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 10
                    }
                    async with session.post(url, json=test_payload) as response:
                        endpoint_status.append({
                            "url": url,
                            "status": "available" if response.status in [200, 422] else "error",
                            "http_status": response.status
                        })
            except Exception as e:
                endpoint_status.append({
                    "url": url,
                    "status": "unavailable",
                    "error": str(e)
                })
        
        return {
            "primary_endpoint": client.api_url,
            "all_endpoints": endpoint_status,
            "total_endpoints": len(client.possible_urls)
        }
        
    except Exception as e:
        logger.error(f"Endpoint status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Endpoint status check failed: {str(e)}")

async def reset_llm_client_endpoint():
    """Reset LLM client and refresh connections"""
    try:
        reset_llm_client()
        
        # Test new client
        client = get_llm_client()
        test_result = await client.generate_response("Connection test", max_tokens=50)
        
        return {
            "status": "success",
            "message": "LLM client reset successfully",
            "new_api_url": client.api_url,
            "test_successful": bool(test_result and len(test_result) > 5)
        }
        
    except Exception as e:
        logger.error(f"LLM client reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM client reset failed: {str(e)}")

async def get_llm_log_tail(lines: int = 200):
    """Return last N lines of backend log (only available for legacy text-generation-webui mode)."""
    tail = model_manager.get_recent_log_tail(lines)
    return PlainTextResponse(tail)

class SwitchModelRequest(BaseModel):
    model_key: str

async def list_available_models():
    """Return catalog and active model with current phase."""
    status = model_manager.get_status()
    return {
        "models": model_manager.list_models(),
        "active_model": status["active_model"],
        "phase": status["phase"],
        "error": status.get("error")
    }

async def switch_model_endpoint(request: SwitchModelRequest):
    """Switch the active local model (Ollama or legacy text-generation-webui)."""
    try:
        result = await model_manager.switch_model(request.model_key)

        if result.get("success"):
            # Reset LLM client to pick up new port
            reset_llm_client()

        return result
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))