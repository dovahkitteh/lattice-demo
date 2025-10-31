#!/usr/bin/env python3
"""
Test script to verify LLM endpoint connectivity for Lattice.
This can be used standalone or integrated into startup scripts.
"""

import asyncio
import aiohttp
import json
import sys
import os
import time
from typing import Optional


async def test_health_endpoint(base_url: str = "http://127.0.0.1:8080") -> bool:
    """Test if the Lattice health endpoint is responding."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print(f"✅ Health endpoint responding: {base_url}/health")
                    return True
                else:
                    print(f"❌ Health endpoint returned status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False


async def test_llm_endpoint(base_url: str = "http://127.0.0.1:8080") -> bool:
    """Test if the LLM chat endpoint is working."""
    try:
        test_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "Connection test"}],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    try:
                        result = await response.json()
                        # Check if we got a proper response structure
                        if "choices" in result or "content" in result:
                            print(f"✅ LLM endpoint responding properly: {base_url}/v1/chat/completions")
                            return True
                        else:
                            print(f"❌ LLM endpoint returned unexpected format: {result}")
                            return False
                    except json.JSONDecodeError as e:
                        print(f"❌ LLM endpoint returned invalid JSON: {e}")
                        return False
                else:
                    response_text = await response.text()
                    print(f"❌ LLM endpoint returned status {response.status}: {response_text[:200]}")
                    return False
    except asyncio.TimeoutError:
        print(f"❌ LLM endpoint test timed out")
        return False
    except Exception as e:
        print(f"❌ LLM endpoint test failed: {e}")
        return False


async def test_specific_endpoints() -> bool:
    """Test specific LLM backend endpoints that Lattice might be trying to connect to."""
    endpoints_to_test = [
        "http://127.0.0.1:11434/v1/chat/completions", # Ollama OpenAI-compatible (primary)
        "http://127.0.0.1:11434/api/chat",             # Ollama native
        "http://127.0.0.1:5000/v1/chat/completions",  # text-generation-webui
        "http://127.0.0.1:7860/v1/chat/completions",  # text-generation-webui web
        "http://127.0.0.1:8000/v1/chat/completions",  # Alternative port
    ]
    
    print("\n🔍 Testing individual LLM endpoints...")
    any_working = False
    
    for endpoint in endpoints_to_test:
        try:
            test_payload = {
                "model": "test",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            
            # Special handling for Ollama native endpoint
            if "/api/chat" in endpoint:
                test_payload = {
                    "model": os.getenv("OLLAMA_MODEL", "test"),
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False
                }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(endpoint, json=test_payload) as response:
                    if response.status == 200:
                        print(f"  ✅ {endpoint}")
                        any_working = True
                    else:
                        print(f"  ❌ {endpoint} (HTTP {response.status})")
        except Exception as e:
            print(f"  ❌ {endpoint} ({type(e).__name__})")
    
    return any_working


async def wait_for_lattice(base_url: str = "http://127.0.0.1:8080", max_wait: int = 60) -> bool:
    """Wait for Lattice to become available."""
    print(f"⏳ Waiting for Lattice to start at {base_url} (max {max_wait}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if await test_health_endpoint(base_url):
            return True
        await asyncio.sleep(2)
    
    return False


async def main():
    """Main test function."""
    print("🧪 Lattice LLM Connectivity Test")
    print("=" * 40)
    
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    
    # Test 1: Health endpoint
    print(f"\n1️⃣ Testing Lattice health endpoint...")
    if not await test_health_endpoint(base_url):
        print("❌ Health endpoint not responding. Is Lattice running?")
        return False
    
    # Test 2: LLM endpoint
    print(f"\n2️⃣ Testing LLM chat endpoint...")
    if not await test_llm_endpoint(base_url):
        print("❌ LLM endpoint not working properly")
        
        # Test 3: Check individual endpoints
        if not await test_specific_endpoints():
            print("\n❌ No working LLM endpoints found!")
            print("\n🔧 Troubleshooting suggestions:")
            print("  1. Ensure Ollama is running: ollama serve")
            print("  2. Check if any LLM service is available on the expected ports")
            print("  3. Verify LLM_API environment variable points to correct endpoint")
            print("  4. Check Lattice logs for connection errors")
            return False
        else:
            print("\n⚠️  Some LLM endpoints are working, but Lattice may have configuration issues")
            return False
    
    print(f"\n✅ All tests passed! Lattice is ready at {base_url}")
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
