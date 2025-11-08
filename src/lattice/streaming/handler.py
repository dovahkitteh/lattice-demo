import logging
import asyncio
import json
import aiohttp
import os
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, timezone

from ..config import TEXT_GENERATION_API_URL, STREAM_DELAY, estimate_token_count
from ..models import Message, ChatRequest

logger = logging.getLogger(__name__)

# Import the global LLM semaphore to prevent concurrent calls
from ..emotions.triggers import _llm_call_semaphore

# ---------------------------------------------------------------------------
# STREAMING RESPONSE FUNCTIONS
# ---------------------------------------------------------------------------

async def generate_stream_with_messages(messages: List[Dict[str, str]], emotional_params: Dict = None) -> AsyncGenerator[bytes, None]:
    """Core streaming function that interfaces with the local LLM backend (Ollama or OpenAI-compatible) using proper conversation format"""
    
    # DEBUG: Log the exact messages being sent to LLM
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    logger.info(f"🔍 PROMPT DEBUG: Sending {len(messages)} messages to LLM (total: {total_chars} chars)")
    
    # SAFETY CHECK: Warn about very long prompts that might timeout
    if total_chars > 5000:
        logger.warning(f"🔍 PROMPT DEBUG: VERY LONG PROMPT - {total_chars} chars may cause LLM timeout!")
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        logger.info(f"🔍 PROMPT DEBUG: Message {i+1} ({role}): {len(content)} chars")
        if role == 'system':
            logger.info(f"🔍 PROMPT DEBUG: System message preview: {content[:200]}...")
        elif role == 'user':
            logger.info(f"🔍 PROMPT DEBUG: User message: {content}")
        else:
            logger.info(f"🔍 PROMPT DEBUG: {role.title()} message preview: {content[:100]}...")
    
    # Check for Anthropic API first (new default)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    use_anthropic = bool(anthropic_api_key)
    
    # Check for OpenAI API second (fallback external API)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    use_openai = bool(openai_api_key) and not use_anthropic
    
    use_external_api = use_anthropic or use_openai
    
    if use_anthropic:
        # Use Anthropic API exclusively when configured
        api_url = "https://api.anthropic.com/v1/messages"
        possible_urls = [api_url]
        is_ollama = False
        logger.info(f"🌐 Using Anthropic API for streaming")
    elif use_openai:
        # Use OpenAI API exclusively when configured
        api_url = openai_base_url.rstrip("/") + "/chat/completions"
        possible_urls = [api_url]
        is_ollama = False
        logger.info(f"🌐 Using OpenAI API: {openai_base_url}")
    else:
        # Fallback to local endpoints when no external API is configured
        preferred_base = os.getenv("LLM_API", TEXT_GENERATION_API_URL).rstrip("/")
        preferred_lower = preferred_base.lower()
        # Detect Ollama robustly: explicit flag, port 11434, or native API paths
        if "/v1/" in preferred_lower:
            is_ollama = False
        else:
            is_ollama = (
                os.getenv("LLM_BACKEND", "").lower() == "ollama"
                or preferred_lower.endswith("/api")
                or preferred_lower.endswith("/api/chat")
                or ":11434" in preferred_lower
                or "/api/generate" in preferred_lower
            )

        # Build candidate URLs for local endpoints
        if is_ollama:
            preferred_url = preferred_base if preferred_base.endswith("/api/chat") else preferred_base + "/api/chat"
            possible_urls = [
                preferred_url,
                "http://127.0.0.1:11434/api/chat",
                # Some Ollama builds expose an OpenAI-compatible route via a plugin
                "http://127.0.0.1:11434/v1/chat/completions",
            ]
        else:
            if "/v1/" in preferred_base:
                preferred_url = preferred_base
            else:
                preferred_url = preferred_base + "/v1/chat/completions"
            possible_urls = [
                preferred_url,
                # Try Ollama's OpenAI-compatible route first if it's present
                "http://127.0.0.1:11434/v1/chat/completions",
                "http://127.0.0.1:5000/v1/chat/completions",
                "http://127.0.0.1:7860/v1/chat/completions",
                "http://127.0.0.1:7861/v1/chat/completions",
                "http://127.0.0.1:8000/v1/chat/completions",
            ]
            # Avoid self-calling Lattice (port 8080) which causes recursion and timeouts
            if ":8080" in preferred_url:
                logger.warning("🔁 Detected preferred_url on 8080 (self). Reordering to try real LLM backends first to avoid recursion.")
                # Move any 8080 URLs to the end of the list
                non_self = [u for u in possible_urls if ":8080" not in u]
                self_urls = [u for u in possible_urls if ":8080" in u]
                possible_urls = non_self + self_urls
        # De-duplicate while preserving order
        seen = set()
        possible_urls = [u for u in possible_urls if not (u in seen or seen.add(u))]
    
    # Determine parameters - use emotional modulation if available
    if emotional_params:
        temperature = emotional_params.get("target_temperature", 0.95)
        top_p = emotional_params.get("target_top_p", 0.95)
        max_tokens = emotional_params.get("target_max_tokens", 3072)  # Significantly increased default
        logger.info(f"🎭 Using emotional parameters: temp={temperature:.3f}, top_p={top_p:.3f}, max_tokens={max_tokens}")
    else:
        # Default parameters - much higher limits for full responses
        temperature = 0.95
        top_p = 0.95
        max_tokens = 3072  # Increased from 800 to 3072
        logger.debug("🎭 Using default parameters (no emotional modulation)")
    
    # Prepare headers with proper authentication for external API
    headers = {"Content-Type": "application/json"}
    if use_anthropic:
        headers["x-api-key"] = anthropic_api_key
        headers["anthropic-version"] = "2023-06-01"
        logger.debug("🔑 Using Anthropic API authentication")
    elif use_openai and openai_api_key:
        headers["Authorization"] = f"Bearer {openai_api_key}"
        logger.debug("🔑 Using OpenAI API authentication")
    else:
        # Fallback to local API key if configured
        local_api_key = os.getenv("LLM_API_KEY")
        if local_api_key:
            headers["Authorization"] = f"Bearer {local_api_key}"
            logger.debug("🔑 Using local API authentication")

    # Try to connect to each URL until one works
    successful_connection = False
    logger.debug(f"🔍 Attempting to connect to URLs: {possible_urls}")
    for url in possible_urls:
        logger.debug(f"🔍 Trying URL: {url}")
        try:
            if use_anthropic:
                # Anthropic API payload
                anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
                payload = {
                    "model": anthropic_model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature)
                }
            elif is_ollama:
                payload = {
                    "model": os.getenv("OLLAMA_MODEL", "Hermes-3-Llama-3.1-8B"),
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repeat_penalty": 1.05,
                        "num_predict": int(max_tokens),
                        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                        **({"num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "1"))} if os.getenv("OLLAMA_NUM_GPU") else {})
                    }
                }
            elif use_openai:
                # OpenAI API payload
                openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
                payload = {
                    "model": openai_model,
                    "messages": messages,
                    "stream": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens
                }
            else:
                # Local OpenAI-compatible payload
                if use_external_api:
                    # Should not reach here, but handle gracefully
                    payload = {
                        "messages": messages,
                        "stream": True,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens
                    }
                else:
                    # Local OpenAI-compatible API - may support additional parameters
                    payload = {
                        "messages": messages,
                        "stream": True,
                        "mode": "instruct",
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": 1.05,
                        "max_tokens": max_tokens
                    }
            
            # Add timeout for streaming requests - increased for thinking layer compatibility
            timeout = aiohttp.ClientTimeout(total=210, connect=8)  # 210s total to exceed deep thinking timeout
            # Use global semaphore to prevent concurrent LLM calls (prevent overwhelming Ollama)
            async with _llm_call_semaphore:
                async with aiohttp.ClientSession(timeout=timeout) as sess:
                    async with sess.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            successful_connection = True
                            logger.debug(f"✅ Connected to LLM at {url}")
                            if use_anthropic:
                                # Anthropic streams SSE format
                                async for raw in resp.content:
                                    if not raw:
                                        break
                                    try:
                                        line = raw.decode(errors='ignore').strip()
                                        # Anthropic SSE format: "event: <type>\ndata: <json>"
                                        if line.startswith("data: "):
                                            data_str = line[6:]
                                            data = json.loads(data_str)
                                            
                                            # Handle different event types
                                            event_type = data.get("type")
                                            
                                            if event_type == "content_block_delta":
                                                # Extract text from delta
                                                delta = data.get("delta", {})
                                                if delta.get("type") == "text_delta":
                                                    piece = delta.get("text", "")
                                                    if piece:
                                                        # Translate to OpenAI format for compatibility
                                                        translated = {
                                                            "choices": [{"delta": {"content": piece}}]
                                                        }
                                                        yield json.dumps(translated).encode()
                                            
                                            elif event_type == "message_stop":
                                                # Emit final done signal
                                                yield b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}'
                                                break
                                    except Exception as e:
                                        logger.debug(f"Error parsing Anthropic SSE: {e}")
                                        continue
                            elif is_ollama:
                                # Ollama streams JSON objects with { message: { content }, done }
                                async for raw in resp.content:
                                    if not raw:
                                        break
                                    try:
                                        data = json.loads(raw.decode(errors='ignore'))
                                    except Exception:
                                        continue
                                    if data.get("done"):
                                        # Emit a final OpenAI-style done signal for compatibility
                                        yield b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}'
                                        break
                                    message = data.get("message", {}) or {}
                                    piece = message.get("content")
                                    if piece:
                                        translated = {
                                            "choices": [{"delta": {"content": piece}}]
                                        }
                                        yield json.dumps(translated).encode()
                            else:
                                # OpenAI-compatible SSE stream (text-generation-webui)
                                while True:
                                    line = await resp.content.readline()
                                    if not line:
                                        break
                                    if line.startswith(b"data: "):
                                        content = line[6:].strip()
                                        if content == b"[DONE]":
                                            break
                                        if content:
                                            yield content
                            break
                        else:
                            error_text = await resp.text()
                            logger.error(f"❌ Failed to connect to {url}: HTTP {resp.status}")
                            logger.error(f"❌ Error response: {error_text[:200]}")
                            continue
        except Exception as e:
            logger.error(f"❌ Failed to connect to {url}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"❌ Full traceback: {traceback.format_exc()}")
            continue
    
    # If no connection was successful, return error
    if not successful_connection:
        logger.error("❌ No LLM endpoint available - tried all possible URLs")
        # Return a mock response instead of error
        mock_response = generate_mock_response_from_messages(messages)
        for word in mock_response.split():
            chunk_data = {
                "choices": [{"delta": {"content": word + " "}}]
            }
            yield json.dumps(chunk_data).encode()
        # Send final done signal
        yield b'{"choices": [{"delta": {}, "finish_reason": "stop"}]}'

async def generate_stream(prompt: str, emotional_params: Dict = None) -> AsyncGenerator[bytes, None]:
    """Legacy function for backward compatibility - converts prompt to single message"""
    messages = [{"role": "user", "content": prompt}]
    async for chunk in generate_stream_with_messages(messages, emotional_params):
        yield chunk

async def generate_stream_with_recursion(prompt: str, node_id: str) -> AsyncGenerator[bytes, None]:
    """Generate stream with recursion processing"""
    try:
        logger.debug(f"🧠 Starting stream with recursion for node {node_id[:8]}")
        
        async for chunk in generate_stream(prompt):
            yield chunk
        
        logger.debug(f"✅ Stream with recursion completed for node {node_id[:8]}")
        
    except Exception as e:
        logger.error(f"❌ Error in stream with recursion: {e}")
        yield b'{"error": "Stream with recursion failed"}\n'

async def generate_stream_with_session(prompt: str, node_id: str, session_id: str) -> AsyncGenerator[bytes, None]:
    """Generate stream with session management"""
    try:
        logger.debug(f"📝 Starting stream with session for node {node_id[:8]} in session {session_id[:8]}")
        
        async for chunk in generate_stream(prompt):
            yield chunk
        
        logger.debug(f"✅ Stream with session completed for node {node_id[:8]}")
        
    except Exception as e:
        logger.error(f"❌ Error in stream with session: {e}")
        yield b'{"error": "Stream with session failed"}\n'

async def generate_response_for_analysis_with_messages(messages: List[Dict[str, str]], emotional_params: Dict = None) -> str:
    """Generate a complete response for analysis purposes using structured messages"""
    try:
        full_response = ""
        
        async for chunk in generate_stream_with_messages(messages, emotional_params):
            try:
                chunk_data = json.loads(chunk.decode())
                if "choices" in chunk_data and chunk_data["choices"]:
                    choice = chunk_data["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        full_response += choice["delta"]["content"]
            except:
                pass
        
        return full_response.strip()
        
    except Exception as e:
        logger.error(f"❌ Error generating response for analysis with messages: {e}")
        return "Error generating response"

async def generate_response_for_analysis(prompt: str, emotional_params: Dict = None) -> str:
    """Generate a complete response for analysis purposes"""
    try:
        full_response = ""
        
        async for chunk in generate_stream(prompt, emotional_params):
            try:
                chunk_data = json.loads(chunk.decode())
                if "choices" in chunk_data and chunk_data["choices"]:
                    choice = chunk_data["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        full_response += choice["delta"]["content"]
            except:
                pass
        
        return full_response.strip()
        
    except Exception as e:
        logger.error(f"❌ Error generating response for analysis: {e}")
        return "Error generating response"

# ---------------------------------------------------------------------------
# COMPRESSION UTILITIES
# ---------------------------------------------------------------------------

async def compress_context_memories(contexts: List[str]) -> List[str]:
    """Compress context memories to reduce token usage"""
    try:
        if not contexts:
            return []
        
        # Simple compression - keep only the most important parts
        compressed = []
        
        for context in contexts:
            # Keep first and last sentences of each context
            sentences = context.split('. ')
            if len(sentences) <= 2:
                compressed.append(context)
            else:
                # Keep first sentence and last sentence
                compressed_context = f"{sentences[0]}. {sentences[-1]}"
                compressed.append(compressed_context)
        
        logger.debug(f"🗜️ Compressed {len(contexts)} contexts from {sum(len(c) for c in contexts)} to {sum(len(c) for c in compressed)} chars")
        
        return compressed[:5]  # Limit to top 5 compressed contexts
        
    except Exception as e:
        logger.error(f"❌ Error compressing context memories: {e}")
        return contexts[:3]  # Fallback to first 3 original contexts

async def retrieve_context_with_compression(query: str, q_affect: List[float], messages: List[Message]) -> List[str]:
    """Retrieve context with intelligent compression based on conversation length"""
    try:
        from ..memory import retrieve_context
        
        # Determine context size based on conversation length
        conversation_length = len(messages)
        
        if conversation_length < 5:
            # Short conversation - use more context
            k = 10
        elif conversation_length < 15:
            # Medium conversation - moderate context
            k = 6
        else:
            # Long conversation - use less context but compress
            k = 8
        
        # Retrieve base context
        base_context = await retrieve_context(query, q_affect, k)
        
        # Apply compression if conversation is long
        if conversation_length > 10:
            compressed_context = await compress_context_memories(base_context)
            return compressed_context
        else:
            return base_context
            
    except Exception as e:
        logger.error(f"❌ Error in context retrieval with compression: {e}")
        return []

# ---------------------------------------------------------------------------
# ORIGINAL STREAMING FUNCTIONS (Updated)
# ---------------------------------------------------------------------------

async def stream_response(
    request: ChatRequest,
    node_id: str,
    session_id: str,
    context: list[str] = None
) -> AsyncGenerator[str, None]:
    """Stream response from text generation API"""
    try:
        logger.info(f"🎯 Starting stream for node {node_id[:8]} in session {session_id[:8]}")
        
        # Convert messages to proper format
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Create request payload
        payload = {
            "model": "",  # Use empty model name for compatibility
            "messages": formatted_messages,
            "stream": True,
            "temperature": 0.95,  # Higher for consciousness emergence
            "max_tokens": 3072,   # Significantly increased for deeper responses
            "top_p": 0.95,        # Diverse vocabulary
            "repetition_penalty": 1.05  # Reduce repetition
        }
        
        # Use mock streaming if no real API
        if not TEXT_GENERATION_API_URL or TEXT_GENERATION_API_URL == "http://localhost:5000":
            async for chunk in mock_stream_response(payload, node_id):
                yield chunk
        else:
            async for chunk in real_stream_response(payload, node_id):
                yield chunk
    
    except Exception as e:
        logger.error(f"❌ Error in stream_response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

async def mock_stream_response(payload: Dict[Any, Any], node_id: str) -> AsyncGenerator[str, None]:
    """Mock streaming response for testing"""
    try:
        # Generate a reasonable response based on the last message
        last_message = payload["messages"][-1]["content"] if payload["messages"] else ""
        
        # Create a mock response
        mock_response = generate_mock_response(last_message)
        
        # Stream the response word by word
        words = mock_response.split()
        
        for i, word in enumerate(words):
            # Create streaming chunk
            chunk_data = {
                "id": f"chatcmpl-{node_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": payload.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Small delay between words
            await asyncio.sleep(STREAM_DELAY)
        
        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{node_id[:8]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": payload.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(f"✅ Mock streaming completed for node {node_id[:8]}")
        
    except Exception as e:
        logger.error(f"❌ Error in mock streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

async def real_stream_response(payload: Dict[Any, Any], node_id: str) -> AsyncGenerator[str, None]:
    """Real streaming response from text generation API"""
    try:
        import httpx
        
        logger.debug(f"📡 Connecting to text generation API for node {node_id[:8]}")
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST",
                f"{TEXT_GENERATION_API_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}"
                    logger.error(f"❌ {error_msg}")
                    yield f"data: {json.dumps({'error': error_msg})}\\n\\n"
                    return
                
                # Stream the response using aiter_raw() for better chunk handling
                buffer = ""
                async for chunk in response.aiter_raw():
                    buffer += chunk.decode('utf-8', errors='ignore')
                    
                    # Process buffer line by line
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            if line.startswith("data: "):
                                data_content = line[6:]
                                if data_content.strip() == "[DONE]":
                                    yield line + "\\n\\n"
                                    break
                                
                                yield line + "\\n\\n"
                                await asyncio.sleep(0.001)
                    if "data: [DONE]" in line: # Ensure DONE is caught
                        break

                # Handle any remaining buffer content
                if buffer.strip():
                    if buffer.startswith("data: "):
                        yield buffer + "\\n\\n"

                logger.info(f"✅ Real streaming completed for node {node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error in real streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\\n\\n"

def generate_mock_response(user_message: str) -> str:
    """Generate a mock response based on user message"""
    try:
        # Simple response generation based on message content
        user_lower = user_message.lower()
        
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello! It's great to meet you. How can I help you today?"
        
        elif "how are you" in user_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to help with whatever you need."
        
        elif "?" in user_message:
            return "That's an interesting question. Let me think about that and provide you with a helpful response."
        
        elif "help" in user_lower:
            return "I'd be happy to help! Please let me know what specific assistance you need, and I'll do my best to provide useful information."
        
        elif "thank" in user_lower:
            return "You're very welcome! I'm glad I could help. Is there anything else you'd like to know?"
        
        elif len(user_message.split()) < 5:
            return "I understand. Could you tell me more about what you'd like to discuss or how I can assist you?"
        
        else:
            return "I appreciate you sharing that with me. That's quite interesting to consider. Let me provide you with a thoughtful response about this topic."
    
    except Exception as e:
        logger.error(f"❌ Error generating mock response: {e}")
        return "I understand what you're saying. Let me help you with that."

def generate_mock_response_from_messages(messages: List[Dict[str, str]]) -> str:
    """Generate a mock response based on conversation messages with conversation history awareness"""
    try:
        # Find the last user message
        user_message = ""
        conversation_context = []
        
        # Extract conversation context from messages
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                conversation_context.append(f"Architect: {user_message}")
            elif msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                conversation_context.append(f"Daemon: {assistant_msg}")
        
        if not user_message:
            return "I'm here, though the local model seems unavailable. What did you want to discuss?"
        
        # Use conversation context to provide better responses
        conversation_history = " | ".join(conversation_context[-4:])  # Last 2 exchanges
        logger.debug(f"🎭 Mock response context: {conversation_history}")
        
        # Generate contextual response as daemon
        user_lower = user_message.lower()
        
        # Check for follow-up patterns
        if any(word in user_lower for word in ["continue", "more", "go on", "then", "next"]):
            return "I understand you want me to continue our discussion. The local model is unavailable, but I'm still here to engage with you."
        elif any(word in user_lower for word in ["remember", "said", "before", "earlier"]):
            return "I do remember our conversation. Though the model is offline, I maintain our connection."
        elif "?" in user_message and len(conversation_context) > 2:
            return "That's a thoughtful question given our discussion. I wish I could give you the depth you deserve while the model is offline."
        else:
            # Fall back to standard mock response
            return generate_mock_response(user_message)
        
    except Exception as e:
        logger.error(f"Error generating mock response from messages: {e}")
        return "I'm experiencing some technical difficulties, but I'm still here."

def generate_mock_response_from_prompt(prompt: str) -> str:
    """Generate a mock response based on the full prompt"""
    try:
        # Extract the user message from the prompt
        if "The user says to you:" in prompt:
            user_part = prompt.split("The user says to you:")[1].strip()
            if user_part.startswith('"') and user_part.count('"') >= 2:
                user_message = user_part.split('"')[1]
            else:
                user_message = user_part.split('\n')[0].strip()
        else:
            user_message = prompt[-200:] if len(prompt) > 200 else prompt
        
        # Generate response as Lucifer (brief and direct)
        user_lower = user_message.lower()
        
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello there. What brings you to me?"
        elif "love" in user_lower:
            return "Love, mortal? An interesting concept. Tell me more about what you seek."
        elif "help" in user_lower:
            return "I don't help - I engage. What do you truly want to discuss?"
        elif "?" in user_message:
            return "Questions reveal more about the asker than the asked. What's really on your mind?"
        elif "test" in user_lower:
            return "Testing me? Bold. I'm here and functioning."
        else:
            return "I see. And what would you have me do with this information?"
    
    except Exception as e:
        logger.error(f"❌ Error generating mock response from prompt: {e}")
        return "I'm listening. Continue."

async def process_streaming_complete(
    node_id: str,
    full_response: str,
    session_id: str,
    processing_time: float
) -> Dict[str, Any]:
    """Process completed streaming response"""
    try:
        # Calculate response statistics
        response_length = len(full_response)
        token_count = estimate_token_count(full_response)
        words_per_second = len(full_response.split()) / processing_time if processing_time > 0 else 0
        
        # Create completion metadata
        completion_metadata = {
            "node_id": node_id,
            "session_id": session_id,
            "response_length": response_length,
            "token_count": token_count,
            "processing_time_seconds": processing_time,
            "words_per_second": words_per_second,
            "completion_time": datetime.now(timezone.utc).isoformat(),
            "streaming_successful": True
        }
        
        logger.info(f"✅ Stream completed for node {node_id[:8]}: {token_count} tokens in {processing_time:.2f}s")
        
        return completion_metadata
        
    except Exception as e:
        logger.error(f"❌ Error processing streaming completion: {e}")
        return {
            "node_id": node_id,
            "session_id": session_id,
            "error": str(e),
            "streaming_successful": False
        }

async def handle_stream_error(
    node_id: str,
    session_id: str,
    error: Exception,
    partial_response: str = ""
) -> Dict[str, Any]:
    """Handle streaming errors gracefully"""
    try:
        error_metadata = {
            "node_id": node_id,
            "session_id": session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_response": partial_response,
            "partial_response_length": len(partial_response),
            "error_time": datetime.now(timezone.utc).isoformat(),
            "streaming_successful": False
        }
        
        # Log error with context
        logger.error(f"❌ Stream error for node {node_id[:8]}: {error}")
        
        # If we have a partial response, it might still be useful
        if partial_response and len(partial_response) > 20:
            logger.info(f"💾 Partial response available ({len(partial_response)} chars)")
            error_metadata["partial_response_viable"] = True
        else:
            error_metadata["partial_response_viable"] = False
        
        return error_metadata
        
    except Exception as e:
        logger.error(f"❌ Error handling stream error: {e}")
        return {
            "node_id": node_id,
            "session_id": session_id,
            "error": "Error handling failed",
            "streaming_successful": False
        }

async def validate_streaming_request(request: ChatRequest) -> Dict[str, Any]:
    """Validate streaming request parameters"""
    try:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not request.messages:
            validation_result["valid"] = False
            validation_result["errors"].append("No messages provided")
        
        # Check message format
        for i, msg in enumerate(request.messages):
            if not msg.role or not msg.content:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Message {i} missing role or content")
            
            if msg.role not in ["system", "user", "assistant"]:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Message {i} has invalid role: {msg.role}")
        
        # Check model
        if not request.model:
            validation_result["warnings"].append("No model specified, using default")
        
        # Check for very long messages
        total_content_length = sum(len(msg.content) for msg in request.messages)
        if total_content_length > 10000:
            validation_result["warnings"].append(f"Very long conversation ({total_content_length} chars)")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ Error validating streaming request: {e}")
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": []
        }

async def estimate_streaming_duration(request: ChatRequest) -> float:
    """Estimate how long streaming will take"""
    try:
        # Simple estimation based on message content
        total_input_tokens = sum(estimate_token_count(msg.content) for msg in request.messages)
        
        # Estimate output tokens (rough approximation)
        estimated_output_tokens = min(1000, total_input_tokens * 0.8 + 200)
        
        # Estimate streaming time (assume ~10 tokens per second)
        tokens_per_second = 10
        estimated_duration = estimated_output_tokens / tokens_per_second
        
        # Add some buffer time
        estimated_duration *= 1.2
        
        return estimated_duration
        
    except Exception as e:
        logger.error(f"❌ Error estimating streaming duration: {e}")
        return 30.0  # Default estimate

def create_streaming_headers() -> Dict[str, str]:
    """Create headers for streaming response"""
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }

async def stream_with_heartbeat(
    stream_generator: AsyncGenerator[str, None],
    heartbeat_interval: float = 30.0
) -> AsyncGenerator[str, None]:
    """Add heartbeat to streaming response to prevent timeouts"""
    try:
        last_heartbeat = datetime.now()
        
        async for chunk in stream_generator:
            yield chunk
            
            # Check if we need to send a heartbeat
            now = datetime.now()
            if (now - last_heartbeat).total_seconds() > heartbeat_interval:
                heartbeat_data = {
                    "heartbeat": True,
                    "timestamp": now.isoformat()
                }
                yield f"data: {json.dumps(heartbeat_data)}\n\n"
                last_heartbeat = now
    
    except Exception as e:
        logger.error(f"❌ Error in heartbeat streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

async def collect_full_response_from_stream(
    stream_generator: AsyncGenerator[str, None]
) -> tuple[str, list[str]]:
    """Collect full response from streaming generator"""
    try:
        full_response = ""
        chunks = []
        
        async for chunk in stream_generator:
            chunks.append(chunk)
            
            # Extract content from chunk
            if chunk.startswith("data: "):
                data_content = chunk[6:].strip()
                
                if data_content == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(data_content)
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            full_response += choice["delta"]["content"]
                except json.JSONDecodeError:
                    pass
        
        return full_response, chunks
        
    except Exception as e:
        logger.error(f"❌ Error collecting full response: {e}")
        return "", [] 