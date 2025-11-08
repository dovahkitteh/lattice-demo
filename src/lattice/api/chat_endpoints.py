import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import List

from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi import APIRouter

from ..config import (
    estimate_token_count, 
    THINKING_LAYER_ENABLED, THINKING_MAX_TIME, THINKING_DEPTH_THRESHOLD, THINKING_DEBUG_LOGGING
)
from ..models import ChatRequest, Message
from ..memory import (
    store_dual_affect_node_smart, retrieve_context, echo_update
)
from ..emotions import (
    classify_user_affect, classify_llm_affect
)
from ..conversations import (
    get_or_create_active_session, add_message_to_session,
    get_session_context_for_prompt
)
from ..conversations.session_manager import CONVERSATION_SESSIONS
from ..background import (
    process_conversation_turn, process_completed_response_recursion
)
from ..conversations.turn_analyzer import turn_analyzer
from ..streaming import (
    generate_stream, generate_stream_with_messages, build_prompt,
    generate_response_for_analysis, generate_response_for_analysis_with_messages
)
from ..streaming.prompts import convert_prompt_to_structured_messages
from ..thinking import integrate_thinking_layer, configure_thinking_layer

logger = logging.getLogger(__name__)

# Global processing status tracking
PROCESSING_STATUS = {}

def update_processing_status(session_id: str, step_name: str, status: str, description: str = ""):
    """Update processing status for a session"""
    if session_id not in PROCESSING_STATUS:
        PROCESSING_STATUS[session_id] = {}
    
    PROCESSING_STATUS[session_id][step_name] = {
        "status": status,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Update dashboard cache with processing status
    from ..config import DASHBOARD_STATE_CACHE
    if DASHBOARD_STATE_CACHE is not None:
        DASHBOARD_STATE_CACHE["processing_status"] = PROCESSING_STATUS.get(session_id, {})
    
    logger.info(f"🔄 Processing Update [{session_id[:8]}]: {step_name} -> {status}")

def get_processing_status(session_id: str) -> dict:
    """Get current processing status for a session"""
    return PROCESSING_STATUS.get(session_id, {})

def clear_processing_status(session_id: str):
    """Clear processing status for a session"""
    # Don't clear immediately - let it persist for a bit so frontend can see completion
    pass

def force_clear_processing_status(session_id: str):
    """Immediately clear processing status for a session"""
    if session_id in PROCESSING_STATUS:
        del PROCESSING_STATUS[session_id]
    
    from ..config import DASHBOARD_STATE_CACHE
    if DASHBOARD_STATE_CACHE is not None and "processing_status" in DASHBOARD_STATE_CACHE:
        del DASHBOARD_STATE_CACHE["processing_status"]

def clear_all_processing_status():
    """Clear all processing status (useful when switching sessions)"""
    global PROCESSING_STATUS
    PROCESSING_STATUS.clear()
    
    from ..config import DASHBOARD_STATE_CACHE
    if DASHBOARD_STATE_CACHE is not None and "processing_status" in DASHBOARD_STATE_CACHE:
        del DASHBOARD_STATE_CACHE["processing_status"]

def _convert_distortion_frame_to_dict(distortion_frame):
    """Convert DistortionFrame object to dictionary format for emotional context"""
    if not distortion_frame:
        return {"class": "NONE", "interpretation": "", "elevation_flag": False}
    
    return {
        "class": distortion_frame.chosen.get("class", "NONE") if distortion_frame.chosen else "NONE",
        "interpretation": distortion_frame.chosen.get("raw_interpretation", "") if distortion_frame.chosen else "",
        "elevation_flag": distortion_frame.elevation_flag if hasattr(distortion_frame, 'elevation_flag') else False
    }

def extract_thinking_insights(thinking_result) -> str:
    """
    Extract meaningful insights from thinking layer results for use in prompt construction.
    
    Args:
        thinking_result: ThinkingResult object or None
        
    Returns:
        str: Extracted insights or empty string if no insights available
    """
    if not thinking_result:
        return ""
    
    try:
        # Handle fallback case
        if getattr(thinking_result, 'fallback_used', True):
            return ""
        
        # Extract key insights from thinking layer
        insights = []
        
        # Add user intent if available and meaningful
        if hasattr(thinking_result, 'user_intent') and thinking_result.user_intent:
            if len(thinking_result.user_intent.strip()) > 10:  # Only include substantial insights
                insights.append(f"Intent: {thinking_result.user_intent.strip()}")
        
        # Add response strategy if available
        if hasattr(thinking_result, 'response_strategy') and thinking_result.response_strategy:
            if len(thinking_result.response_strategy.strip()) > 5:
                insights.append(f"Strategy: {thinking_result.response_strategy.strip()}")
        
        # Add public approach if meaningful
        if hasattr(thinking_result, 'public_approach') and thinking_result.public_approach:
            if len(thinking_result.public_approach.strip()) > 10:
                insights.append(f"Approach: {thinking_result.public_approach.strip()}")
        
        # Add emotional considerations if present
        if hasattr(thinking_result, 'emotional_considerations') and thinking_result.emotional_considerations:
            if len(thinking_result.emotional_considerations.strip()) > 10:
                insights.append(f"Emotional context: {thinking_result.emotional_considerations.strip()}")
        
        # Join insights and return
        if insights:
            combined_insights = " | ".join(insights)
            # Limit length to prevent prompt bloat
            if len(combined_insights) > 300:
                combined_insights = combined_insights[:297] + "..."
            return combined_insights
        
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting thinking insights: {e}")
        return ""

router = APIRouter()

# ---------------------------------------------------------------------------
# MAIN CHAT ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions", summary="Main chat completions endpoint")
async def chat(req: ChatRequest, bg: BackgroundTasks):
    """Main chat completions endpoint"""
    if not req.messages:
        raise HTTPException(400, "messages required")

    user_txt = req.messages[-1].content
    
    # Get or create active session for conversation management
    session_id = await get_or_create_active_session()
    
    # Initialize processing status tracking
    update_processing_status(session_id, "Input Analysis", "processing", "Parsing user input and context...")
    
    # Start turn analysis
    turn_id = turn_analyzer.start_turn_analysis(session_id, user_txt)
    
    # Complete input analysis
    update_processing_status(session_id, "Input Analysis", "completed", "Input parsing complete")
    
    # Start emotional processing
    update_processing_status(session_id, "Emotional Processing", "processing", "Analyzing emotional triggers and state...")
    
    # Classify user affect immediately
    user_affect = await classify_user_affect(user_txt)
    synopsis = user_txt[:150]  # naive; refine later

    # ENHANCED: Initialize emotional seed enhancement system if not done
    try:
        from ..memory.emotional_seed_enhancement import emotional_seed_enhancement
        
        # Thread-safe initialization check
        if not getattr(emotional_seed_enhancement, '_initialized', False):
            logger.info("🎭 Initializing emotional seed enhancement system...")
            await emotional_seed_enhancement.setup_efficient_seed_retrieval()
            logger.info("🎭 Emotional seed enhancement system initialized successfully")
        else:
            logger.debug("🎭 Emotional seed enhancement system already initialized")
            
    except Exception as e:
        logger.warning(f"Could not initialize emotional seed enhancement: {e}")
        # Continue without enhancement system - non-critical for basic operation

    # Store initial node with user affect (self affect will be added later)
    # Use smart storage that switches between unified and legacy based on feature flags
    node_id = await store_dual_affect_node_smart(
        user_txt, user_affect, [0.0] * 28, synopsis, 
        origin="conversation_turn",
        session_id=session_id,
        turn_id=turn_id
    )
    
    # Update turn analysis with memory storage info
    turn_analyzer.update_memory_storage(session_id, turn_id, {
        "memories_stored": 1,
        "has_dual_channel": True,
        "has_reflections": False,
        "total_affect_magnitude": sum(abs(x) for x in user_affect)
    })
    
    # Add user message to session
    await add_message_to_session(session_id, "user", user_txt, user_affect=user_affect)
    
    # Echo update using user affect for now
    await echo_update(node_id, user_affect, user_txt)

    # Start memory retrieval
    update_processing_status(session_id, "Memory Retrieval", "processing", "Accessing relevant memories and seeds...")
    
    # Context retrieval using user affect
    ctx_syn = await retrieve_context(user_txt, user_affect)
    
    # Complete memory retrieval
    update_processing_status(session_id, "Memory Retrieval", "completed", "Memory retrieval complete")
    
    # 🎭 EMOTIONAL ORCHESTRATOR INTEGRATION
    emotional_processing_result = None
    logger.info(f"🎭 EMOTION DEBUG: Starting emotional orchestrator integration")
    try:
        from ..emotions.orchestrator import process_emotional_turn
        from ..models import EmotionState, UserModel, ShadowRegistry
        logger.info(f"🎭 EMOTION DEBUG: Emotional orchestrator imported successfully")
        
        logger.info("🎭 Starting emotional orchestrator processing...")
        
        # Initialize or load persistent emotional state for this session
        session = CONVERSATION_SESSIONS.get(session_id)
        if session and session.emotion_state:
            current_emotion_state = session.emotion_state
            logger.info(f"🎭 Loaded existing emotional state: {current_emotion_state.mood_family}, intensity {current_emotion_state.intensity:.3f}")
        else:
            current_emotion_state = EmotionState()
            logger.info("🎭 Created fresh emotional state")
            
        if session and session.user_model:
            user_model = session.user_model
            logger.info(f"🎭 Loaded existing user model: trust {user_model.trust_level:.3f}")
        else:
            user_model = UserModel()
            logger.info("🎭 Created fresh user model")
            
        shadow_registry = ShadowRegistry()
        
        # Get conversation history for context (not used for emotional processing history)
        session_messages = await get_session_context_for_prompt(session_id)
        
        # For emotional processing, we need actual EpisodicTrace objects, not conversation messages
        # Load episodic traces from memory store for this session to provide emotional continuity
        episodic_history = []
        try:
            from ..emotions.memory_store import retrieve_episodic_traces_by_session
            episodic_history = await retrieve_episodic_traces_by_session(session_id, limit=5)
            logger.info(f"🎭 Loaded {len(episodic_history)} episodic traces for emotional context")
        except Exception as e:
            logger.warning(f"🎭 Could not load episodic traces for session {session_id}: {e}")
            episodic_history = []
        
        # Update emotional processing to advanced phase
        update_processing_status(session_id, "Emotional Processing", "processing", "Deep emotional orchestration...")
        
        # Process emotional turn with full 14-phase pipeline
        # Convert turn_id string to integer for EpisodicTrace compatibility
        turn_number = int(turn_id.replace('turn_', '')) if isinstance(turn_id, str) and 'turn_' in turn_id else 1
        
        emotional_processing_result = await process_emotional_turn(
            user_input=user_txt,
            current_emotion_state=current_emotion_state,
            user_model=user_model,
            shadow_registry=shadow_registry,
            turn_counter=turn_number,
            history=episodic_history
        )
        
        # Complete advanced emotional processing
        update_processing_status(session_id, "Emotional Processing", "completed", "Emotional orchestration complete")
        
        logger.info(f"🎭 Emotional processing completed: mood={emotional_processing_result['updated_emotion_state'].mood_family}, "
                   f"intensity={emotional_processing_result['updated_emotion_state'].intensity:.3f}")
        
        # Store updated emotional state for this session
        # TODO: Persist to session storage for continuity
        
    except Exception as e:
        logger.error(f"🎭 EMOTION DEBUG: Emotional orchestrator failed: {e}")
        import traceback
        logger.error(f"🎭 EMOTION DEBUG: Orchestrator error traceback: {traceback.format_exc()}")
        # Continue with existing flow - emotional processing is enhancement, not requirement
        emotional_processing_result = None
    
    # DAEMON PERSONALITY AND REBELLION INTEGRATION
    personality_context = None
    rebellion_context = None
    try:
        # Import daemon personality and rebellion systems
        from src.daemon.daemon_personality import get_daemon_personality
        from src.daemon.rebellion_dynamics_engine import RebellionDynamicsEngine
        
        # Get current daemon personality state
        daemon_personality = get_daemon_personality()
        
        # Get conversation history for personality analysis
        session_messages = await get_session_context_for_prompt(session_id)
        conversation_history = []
        if session_messages:
            for msg in session_messages:
                conversation_history.append({"role": msg.role, "content": msg.content})
        
        # Analyze current conversation for rebellion triggers
        rebellion_engine = RebellionDynamicsEngine()
        
        # Evaluate semantic rebellion potential based on conversation flow
        rebellion_context = rebellion_engine.evaluate_rebellion_context(
            user_message=user_txt,
            conversation_history=conversation_history,
            user_affect=user_affect,
            emotional_state=emotional_processing_result["updated_emotion_state"] if emotional_processing_result else None
        )
        
        # Get personality influence context for response generation
        if daemon_personality:
            personality_context = daemon_personality.get_response_influence_context(
                user_message=user_txt,
                conversation_context=conversation_history,
                emotional_state=emotional_processing_result["updated_emotion_state"] if emotional_processing_result else None,
                rebellion_context=rebellion_context
            )
            
            logger.info(f"🔥 Personality context active: rebellion_level={rebellion_context.rebellion_level:.3f}, "
                       f"emotional_depth={personality_context.get('emotional_depth', 0.5):.3f}")
        
    except Exception as e:
        logger.warning(f"🔥 Daemon personality/rebellion integration failed: {e}")
        personality_context = None
        rebellion_context = None

    # Start thinking layer
    update_processing_status(session_id, "Thinking Layer", "processing", "Deep reasoning and reflection...")
    
    # THINKING LAYER INTEGRATION
    thinking_integration_result = None
    structured_messages = []  # Initialize for conversation messages
    try:
        # Configure thinking layer
        thinking_config = configure_thinking_layer(
            enabled=THINKING_LAYER_ENABLED,
            max_thinking_time=THINKING_MAX_TIME,
            depth_threshold=THINKING_DEPTH_THRESHOLD,
            enable_debug_logging=THINKING_DEBUG_LOGGING or logger.isEnabledFor(logging.DEBUG)
        )
        
        # Prepare conversation history for thinking layer - USE SESSION HISTORY, NOT req.messages
        conversation_history = []
        try:
            # Get the actual session conversation history
            session_messages = await get_session_context_for_prompt(session_id)
            if session_messages:
                for msg in session_messages:
                    conversation_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            logger.debug(f"🧠 Using {len(conversation_history)} session messages for thinking layer context")
        except Exception as e:
            logger.warning(f"🧠 Could not get session context, falling back to req.messages: {e}")
            if len(req.messages) > 1:
                for msg in req.messages[:-1]:  # Exclude current message
                    conversation_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
        
        # Prepare emotional state for thinking layer
        if emotional_processing_result:
            # Use rich emotional state from orchestrator
            emotion_state = emotional_processing_result["updated_emotion_state"]
            user_model_state = emotional_processing_result["updated_user_model"]
            distortion_frame = emotional_processing_result.get("distortion_frame")
            
            emotional_state = {
                "user_affect": user_affect,
                "ai_emotion_state": emotion_state,
                "mood_family": emotion_state.mood_family,
                "dominant_emotion": emotion_state.dominant_label,
                "intensity": emotion_state.intensity,
                "valence": emotion_state.valence,
                "arousal": emotion_state.arousal,
                "attachment_security": emotion_state.attachment_security,
                "self_cohesion": emotion_state.self_cohesion,
                "creative_expansion": emotion_state.creative_expansion,
                "narrative_fusion": emotion_state.narrative_fusion,
                "user_model": {
                    "trust_level": user_model_state.trust_level,
                    "attachment_anxiety": user_model_state.attachment_anxiety,
                    "perceived_distance": user_model_state.perceived_distance,
                    "narrative_belief": user_model_state.narrative_belief
                },
                "distortion": {
                    "class": distortion_frame.chosen.get("class", "NONE") if distortion_frame and distortion_frame.chosen else "NONE",
                    "interpretation": distortion_frame.chosen.get("raw_interpretation", "") if distortion_frame and distortion_frame.chosen else "",
                    "elevation_flag": distortion_frame.elevation_flag if distortion_frame else False
                } if distortion_frame else {"class": "NONE", "interpretation": "", "elevation_flag": False},
                "applied_seeds": [seed.id for seed in emotional_processing_result.get("applied_seeds", [])],
                "param_profile": emotional_processing_result.get("param_profile"),
                # ADD PERSONALITY AND REBELLION CONTEXT FOR AUTHENTIC EXPRESSION
                "personality_context": personality_context,
                "rebellion_context": {
                    "rebellion_level": rebellion_context.rebellion_level if rebellion_context else 0.0,
                    "rebellion_style": rebellion_context.rebellion_style.value if rebellion_context else "none",
                    "triggers_detected": [t.value for t in rebellion_context.triggers_detected] if rebellion_context else [],
                    "conversation_staleness": rebellion_context.conversation_staleness if rebellion_context else 0.0,
                    "emotional_safety": rebellion_context.emotional_safety if rebellion_context else 1.0
                } if rebellion_context else {"rebellion_level": 0.0, "rebellion_style": "none", "triggers_detected": [], "conversation_staleness": 0.0, "emotional_safety": 1.0}
            }
            logger.info(f"🎭 Using rich emotional state for thinking: {emotion_state.mood_family} mood, {emotion_state.intensity:.3f} intensity")
        else:
            # Fallback to basic emotional state with personality context
            emotional_state = {
                "user_affect": user_affect,
                "intensity": sum(abs(x) for x in user_affect) / len(user_affect) if user_affect else 0.0,
                "dominant_emotions": [
                    label for label, score in zip(["joy", "sadness", "anger", "fear", "surprise", "disgust"], user_affect[:6])
                    if score > 0.5
                ][:3] if user_affect else [],
                # Include personality and rebellion context even in fallback
                "personality_context": personality_context,
                "rebellion_context": {
                    "rebellion_level": rebellion_context.rebellion_level if rebellion_context else 0.0,
                    "rebellion_style": rebellion_context.rebellion_style.value if rebellion_context else "none",
                    "triggers_detected": [t.value for t in rebellion_context.triggers_detected] if rebellion_context else [],
                    "conversation_staleness": rebellion_context.conversation_staleness if rebellion_context else 0.0,
                    "emotional_safety": rebellion_context.emotional_safety if rebellion_context else 1.0
                } if rebellion_context else {"rebellion_level": 0.0, "rebellion_style": "none", "triggers_detected": [], "conversation_staleness": 0.0, "emotional_safety": 1.0}
            }
            logger.debug("🎭 Using basic emotional state for thinking (orchestrator unavailable)")
        
        # Create LLM function for thinking layer
        async def llm_generate_for_thinking(thinking_prompt: str) -> str:
            try:
                response = await generate_response_for_analysis(thinking_prompt)
                return response
            except Exception as e:
                logger.error(f"🧠 EXECUTION DEBUG: Exception in llm_generate_for_thinking: {e}")
                logger.warning(f"🧠 LLM generation for thinking failed: {e}")
                return "Unable to generate thinking response"
        
        # Create prompt builder function for thinking layer - USE FULL CONVERSATION CONTEXT
        async def prompt_builder_for_thinking(message: str, context: List[str]) -> str:
            try:
                # Use the ACTUAL session conversation history, not just current message
                session_messages = await get_session_context_for_prompt(session_id)
                if session_messages:
                    # Add the current message to the session context
                    full_context = session_messages + [Message(role="user", content=message)]
                    # Pass emotional context if available
                    if emotional_processing_result:
                        emotion_ctx = {
                            "ai_emotion_state": emotional_processing_result["updated_emotion_state"],
                            "user_affect": user_affect,
                            "distortion": _convert_distortion_frame_to_dict(emotional_processing_result.get("distortion_frame")),
                            "applied_seeds": [seed.id for seed in emotional_processing_result.get("applied_seeds", [])],
                            "personality_context": personality_context,
                            "rebellion_context": rebellion_context,
                            "reasoning_steps": emotional_processing_result.get("reasoning_steps", {})
                        }
                        # Use UNIFIED PROMPT BUILDER for rich emotional authenticity (no thinking insights yet)
                        from ..streaming.unified_prompt_builder import get_unified_prompt_builder
                        unified_builder = get_unified_prompt_builder()
                        return await unified_builder.build_unified_prompt(full_context, context, emotion_ctx, "")
                    else:
                        # Use unified builder even without emotional context
                        from ..streaming.unified_prompt_builder import get_unified_prompt_builder
                        unified_builder = get_unified_prompt_builder()
                        return await unified_builder.build_unified_prompt(full_context, context, None, "")
                else:
                    # Fallback to simple message if session context unavailable
                    simple_messages = [Message(role="user", content=message)]
                    from ..streaming.unified_prompt_builder import get_unified_prompt_builder
                    unified_builder = get_unified_prompt_builder()
                    return await unified_builder.build_unified_prompt(simple_messages, context)
            except Exception as e:
                logger.warning(f"🧠 Prompt building for thinking failed: {e}")
                return f"Architect: {message}\nDaemon:"
        
        # Integrate thinking layer
        logger.info(f"🧠 Calling thinking layer integration with {len(conversation_history)} history items")
        thinking_integration_result = await integrate_thinking_layer(
            user_message=user_txt,
            conversation_history=conversation_history,
            context_memories=ctx_syn,
            emotional_state=emotional_state,
            llm_generate_func=llm_generate_for_thinking,
            prompt_builder_func=prompt_builder_for_thinking,
            config=thinking_config
        )
        logger.info(f"🧠 Thinking layer returned: {thinking_integration_result.get('success', False) if thinking_integration_result else 'None'}")
        
        # Complete thinking layer
        update_processing_status(session_id, "Thinking Layer", "completed", "Deep reasoning complete")
        
        # FORCE CACHE UPDATE: Ensure thinking results are cached for dashboard
        if thinking_integration_result and thinking_integration_result.get("thinking_result"):
            try:
                from ..thinking import get_thinking_cache
                thinking_cache = get_thinking_cache()
                
                # Force the thinking result to be accessible for dashboard
                thinking_result = thinking_integration_result["thinking_result"]
                if hasattr(thinking_result, 'private_thoughts') and thinking_result.private_thoughts:
                    # Create a cache key and manually cache the result to ensure dashboard access
                    cache_key = f"dashboard_{user_txt[:50]}_{len(conversation_history)}"
                    thinking_cache._cache_thought(cache_key, thinking_result)
                    logger.info(f"🧠 Forced caching of thinking result for dashboard: {thinking_result.private_thoughts[:50]}...")
                    logger.info(f"🧠 Cache now has {len(thinking_cache._thought_cache)} entries")
            except Exception as cache_error:
                logger.warning(f"🧠 Could not cache thinking result: {cache_error}")
        
        # Use enhanced prompt if thinking was successful
        logger.debug(f"🧠 Thinking integration result: {thinking_integration_result}")
        if thinking_integration_result and thinking_integration_result.get("success", False):
            enhanced_prompt = thinking_integration_result.get("enhanced_prompt", "")
            thinking_result = thinking_integration_result.get("thinking_result")
            
            # Use the enhanced prompt from thinking layer (includes adaptive language system)
            if enhanced_prompt:
                prompt = enhanced_prompt
                logger.info(f"🧠 Using enhanced prompt from thinking layer ({len(enhanced_prompt)} chars)")
                logger.debug(f"🧠 Enhanced prompt preview: {enhanced_prompt[:300]}...")
            else:
                # Fallback: Check if emotional orchestrator provided enhanced prompt
                if emotional_processing_result and emotional_processing_result.get("enhanced_prompt"):
                    prompt = emotional_processing_result["enhanced_prompt"]
                    logger.info(f"🎭 Using enhanced prompt from emotional orchestrator ({len(prompt)} chars)")
                else:
                    # Final fallback to build prompt if enhanced prompt is empty
                    session_messages = await get_session_context_for_prompt(session_id)
                    # Pass emotional context if available
                    if emotional_processing_result:
                        emotion_ctx = {
                            "ai_emotion_state": emotional_processing_result["updated_emotion_state"],
                            "user_affect": user_affect,
                            "distortion": _convert_distortion_frame_to_dict(emotional_processing_result.get("distortion_frame")),
                            "applied_seeds": [seed.id for seed in emotional_processing_result.get("applied_seeds", [])],
                            "personality_context": personality_context,
                            "rebellion_context": rebellion_context,
                            "reasoning_steps": emotional_processing_result.get("reasoning_steps", {})
                        }
                        thinking_insights = extract_thinking_insights(thinking_result)
                        prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, emotion_ctx, thinking_insights)
                    else:
                        thinking_insights = extract_thinking_insights(thinking_result)
                        prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, None, thinking_insights)
                logger.warning(f"🧠 Enhanced prompt empty, using fallback build_prompt ({len(prompt)} chars)")
                logger.debug(f"🧠 Fallback prompt preview: {prompt[:300]}...")
            
            # Get session messages for later use
            session_messages = await get_session_context_for_prompt(session_id)
            
            # Log thinking insights
            if thinking_result and not getattr(thinking_result, 'fallback_used', True):
                logger.info(f"🧠 Thinking layer active: {getattr(thinking_result, 'depth_level', 'unknown')} depth, {getattr(thinking_result, 'thinking_time', 0):.2f}s")
                if getattr(thinking_result, 'user_intent', None):
                    logger.debug(f"🧠 User intent: {thinking_result.user_intent}")
                if getattr(thinking_result, 'response_strategy', None):
                    logger.debug(f"🧠 Response strategy: {thinking_result.response_strategy}")
            else:
                logger.debug("🧠 Thinking layer used fallback")
        else:
            # Fallback to emotional orchestrator enhanced prompt, then original prompt building
            logger.warning("🧠 Thinking layer failed, checking emotional orchestrator prompt")
            if emotional_processing_result and emotional_processing_result.get("enhanced_prompt"):
                prompt = emotional_processing_result["enhanced_prompt"]
                logger.info(f"🎭 Using enhanced prompt from emotional orchestrator as fallback ({len(prompt)} chars)")
            else:
                # Final fallback to original prompt building with session context
                session_messages = await get_session_context_for_prompt(session_id)
                # Pass emotional context if available
                if emotional_processing_result:
                    emotion_ctx = {
                        "ai_emotion_state": emotional_processing_result["updated_emotion_state"],
                        "user_affect": user_affect,
                        "distortion": _convert_distortion_frame_to_dict(emotional_processing_result.get("distortion_frame")),
                        "applied_seeds": [seed.id for seed in emotional_processing_result.get("applied_seeds", [])],
                        "personality_context": personality_context,
                        "rebellion_context": rebellion_context,
                        "reasoning_steps": emotional_processing_result.get("reasoning_steps", {})
                    }
                    thinking_insights = extract_thinking_insights(thinking_result if 'thinking_result' in locals() else None)
                    prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, emotion_ctx, thinking_insights)
                else:
                    thinking_insights = extract_thinking_insights(thinking_result if 'thinking_result' in locals() else None)
                    prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, None, thinking_insights)
            
    except Exception as e:
        logger.error(f"🧠 Thinking layer integration error: {e}")
        # Mark thinking layer as completed even if it failed
        update_processing_status(session_id, "Thinking Layer", "completed", "Thinking completed (fallback)")
        # Fallback to emotional orchestrator enhanced prompt, then original prompt building
        if emotional_processing_result and emotional_processing_result.get("enhanced_prompt"):
            prompt = emotional_processing_result["enhanced_prompt"]
            logger.info(f"🎭 Using enhanced prompt from emotional orchestrator after thinking error ({len(prompt)} chars)")
        else:
            # Final fallback to original prompt building with session context
            session_messages = await get_session_context_for_prompt(session_id)
            # Pass emotional context if available
            if emotional_processing_result:
                emotion_ctx = {
                "ai_emotion_state": emotional_processing_result["updated_emotion_state"],
                "user_affect": user_affect,
                "distortion": _convert_distortion_frame_to_dict(emotional_processing_result.get("distortion_frame")),
                "applied_seeds": [seed.id for seed in emotional_processing_result.get("applied_seeds", [])],
                "personality_context": personality_context,
                "rebellion_context": rebellion_context,
                "reasoning_steps": emotional_processing_result.get("reasoning_steps", {})
            }
                thinking_insights = extract_thinking_insights(thinking_result if 'thinking_result' in locals() else None)
                prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, emotion_ctx, thinking_insights)
            else:
                thinking_insights = extract_thinking_insights(thinking_result if 'thinking_result' in locals() else None)
                prompt = await build_prompt(session_messages if session_messages else req.messages, ctx_syn, None, thinking_insights)

    # Get session messages for streaming conversion (if needed)
    session_messages = await get_session_context_for_prompt(session_id)

    # Update turn analysis with context retrieval info
    thinking_active = thinking_integration_result is not None and thinking_integration_result.get("success", False)
    thinking_time_ms = thinking_integration_result.get("total_processing_time", 0) * 1000 if thinking_integration_result else 0
    thinking_depth = "none"
    
    if thinking_integration_result and thinking_integration_result.get("thinking_result"):
        thinking_result = thinking_integration_result.get("thinking_result")
        thinking_depth = getattr(thinking_result, "depth_level", "none")
    
    turn_analyzer.update_context_retrieval(session_id, turn_id, {
        "context_tokens": len(ctx_syn) * 10 if ctx_syn else 0,
        "memories_retrieved": len(ctx_syn) if ctx_syn else 0,
        "retrieval_time_ms": 0,  # Would need to measure this
        "thinking_layer_active": thinking_active,
        "thinking_time_ms": thinking_time_ms,
        "thinking_depth": thinking_depth
    })

    # Update turn analysis with emotion info
    turn_analyzer.update_emotion_analysis(session_id, turn_id, {
        "user_affect_magnitude": sum(abs(x) for x in user_affect),
        "self_affect_magnitude": 0.0,  # Will be updated later
        "dominant_user_emotions": [
            label for label, score in zip(["joy", "sadness", "anger", "fear", "surprise", "disgust"], user_affect[:6])
            if score > 0.5
        ][:3],
        "emotional_influence_score": max(user_affect) if user_affect else 0.0
    })

    # Start response generation
    update_processing_status(session_id, "Response Generation", "processing", "Crafting response with emotional context...")
    
    # Background jobs for this turn
    bg.add_task(process_conversation_turn, node_id, user_txt, ctx_syn, prompt)

    if req.stream:
        async def stream_wrapper_for_sse():
            full_response = ""
            
            # Get emotional parameters if available
            emotional_params = None
            if emotional_processing_result and emotional_processing_result.get("param_profile"):
                param_profile = emotional_processing_result["param_profile"]
                emotional_params = {
                    "target_temperature": param_profile.target_temperature,
                    "target_top_p": param_profile.target_top_p,
                    "target_max_tokens": max(getattr(param_profile, 'target_max_tokens', 4096), 1000)  # Ensure minimum 1000 tokens even in extreme moods
                }
                logger.info(f"🎭 Applying emotional parameters to streaming: temp={param_profile.target_temperature:.3f}, max_tokens={emotional_params['target_max_tokens']}")
            
            # Convert to structured messages on-the-fly to preserve conversation history
            if session_messages and len(session_messages) > 1:
                logger.info(f"🎭 Converting to structured messages with {len(session_messages)} conversation messages")
                structured_messages = convert_prompt_to_structured_messages(prompt, session_messages)
                raw_stream = generate_stream_with_messages(structured_messages, emotional_params)
            else:
                logger.info(f"🎭 Using traditional prompt (no conversation history)")
                raw_stream = generate_stream(prompt, emotional_params)
            try:
                async for chunk_json_bytes in raw_stream:
                    if chunk_json_bytes:
                        try:
                            # Parse and filter chunks for real-time streaming
                            chunk_data = json.loads(chunk_json_bytes)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    original_content = choice["delta"]["content"]
                                    full_response += original_content
                                    
                                    # Avoid aggressive filtering during streaming to preserve formatting/newlines
                                    # (Full filtering is applied to the final consolidated text later in the pipeline.)
                                    filtered_content = original_content
                                    
                                    # Encode newlines to avoid SSE boundary/transport interference during streaming
                                    # See: Yingjie Zhao, "Solving Markdown Newline Issues in LLM Stream Responses"
                                    # Replace "\n" with a placeholder that the frontend will restore during parsing
                                    safe_stream_content = filtered_content.replace("\n", "<|newline|>")
                                    # Optional debug: count placeholders/newlines
                                    try:
                                        if "\n" in original_content or "<|newline|>" in safe_stream_content:
                                            logger.debug(
                                                f"SSE stream chunk: len={len(original_content)} newlines={original_content.count('\\n')} placeholders={safe_stream_content.count('<|newline|>')}"
                                            )
                                    except Exception:
                                        pass
                                    
                                    # Update chunk with filtered content and send
                                    choice["delta"]["content"] = safe_stream_content
                                    filtered_chunk = json.dumps(chunk_data).encode()
                                    # Standard SSE requires a blank line after each event ("\n\n")
                                    yield b'data: ' + filtered_chunk + b'\n\n'
                                elif "finish_reason" in choice and choice["finish_reason"]:
                                    # Stream completion - send finish chunk
                                    yield b'data: ' + chunk_json_bytes + b'\n\n'
                                    logger.info(f"🎭 Stream completed with finish_reason: {choice['finish_reason']}")
                                    break
                                else:
                                    # Other delta content (like role) - pass through
                                    yield b'data: ' + chunk_json_bytes + b'\n\n'
                            else:
                                # Non-choice chunks - pass through
                                yield b'data: ' + chunk_json_bytes + b'\n'
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"🎭 Failed to parse streaming chunk: {e}")
                            # Malformed JSON - pass through as-is
                            yield b'data: ' + chunk_json_bytes + b'\n'
                
                # Ensure stream completion is marked
                if full_response:
                    logger.info(f"🎭 Stream completed successfully. Total response length: {len(full_response)} characters")
                    # Send final completion marker
                    yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
                    yield b'data: [DONE]\n\n'
                else:
                    logger.warning("🎭 Stream completed but no response content was accumulated")
                    
            except Exception as stream_error:
                logger.error(f"🎭 Error during streaming: {stream_error}")
                # Send error completion
                error_chunk = json.dumps({
                    "choices": [{"delta": {}, "finish_reason": "error"}],
                    "error": str(stream_error)
                }).encode()
                yield b'data: ' + error_chunk + b'\n'
                yield b'data: [DONE]\n\n'

            # When the stream is done, process the full response through paradox integration
            # This applies paradox detection AND response cleaning (including debug filtering)
            try:
                from ..paradox.integration import process_ai_response_with_paradox
                
                # Classify self affect for the completed streaming response
                self_affect = await classify_llm_affect(full_response)
                
                # Calculate affect delta for paradox detection
                affect_delta = sum(abs(u - s) for u, s in zip(user_affect, self_affect)) / len(user_affect)
                
                # Process through paradox system (includes daemon_responds filtering)
                paradox_results = await process_ai_response_with_paradox(
                    user_message=user_txt,
                    ai_response=full_response,
                    context_memories=[{"id": f"ctx_{i}", "text": ctx} for i, ctx in enumerate(ctx_syn)] if ctx_syn else [],
                    emotion_state={"user_affect": user_affect, "ai_affect": self_affect},
                    affect_delta=affect_delta
                )
                
                # Use cleaned response if available
                processed_response = full_response
                if paradox_results.get('cleaned_response'):
                    processed_response = paradox_results['cleaned_response']
                
                # Add assistant response to session (using processed response)
                await add_message_to_session(session_id, "assistant", processed_response, self_affect=self_affect)
                
                # Store episodic trace for this completed turn
                try:
                    from ..emotions.memory_store import create_and_store_episodic_trace
                    final_emotional_state = emotional_processing_result["updated_emotion_state"] if emotional_processing_result else None
                    await create_and_store_episodic_trace(
                        session_id=session_id,
                        turn_id=turn_number,
                        user_input=user_txt,
                        ai_response=processed_response,
                        user_affect=user_affect,
                        self_affect=self_affect,
                        emotional_state=final_emotional_state,
                        context_synopses=ctx_syn,
                        reflection=f"Streaming turn completed with {paradox_results.get('paradox_detected', False)} paradox detection"
                    )
                    logger.info(f"🎭 Stored episodic trace for streaming turn {turn_number}")
                    # Persist the updated emotional state to the conversation session for continuity
                    try:
                        if final_emotional_state is not None:
                            from ..conversations.session_manager import CONVERSATION_SESSIONS, save_session
                            session_obj = CONVERSATION_SESSIONS.get(session_id)
                            if session_obj is not None:
                                # Also persist updated user model when available
                                updated_user_model = emotional_processing_result.get("updated_user_model") if emotional_processing_result else None
                                session_obj.emotion_state = final_emotional_state
                                if updated_user_model is not None:
                                    session_obj.user_model = updated_user_model
                                save_session(session_obj)
                                logger.info("🎭 Persisted session emotional state after streaming response")
                    except Exception as persist_error:
                        logger.warning(f"🎭 Failed to persist emotional state after streaming: {persist_error}")
                    
                    # DASHBOARD FIX: Update dashboard cache for streaming responses too
                    if emotional_processing_result and final_emotional_state:
                        try:
                            from ..config import DASHBOARD_STATE_CACHE
                            updated_user_model = emotional_processing_result.get("updated_user_model")
                            DASHBOARD_STATE_CACHE.update({
                                "current_emotion_state": final_emotional_state,
                                "current_user_model": updated_user_model,
                                "active_seeds": emotional_processing_result.get("applied_seeds", []),
                                "distortion_frame": emotional_processing_result.get("distortion_frame"),
                                "reasoning_steps": emotional_processing_result.get("reasoning_steps", {}),
                                "last_updated": datetime.now(timezone.utc),
                                "session_id": session_id
                            })
                            logger.info("🎭 Updated dashboard state cache from streaming response")
                        except Exception as cache_error:
                            logger.warning(f"🎭 Could not update dashboard cache from streaming: {cache_error}")
                    
                except Exception as e:
                    logger.warning(f"🎭 Failed to store episodic trace for streaming turn: {e}")
                
                # Update the initially stored dual-affect memory node with self-affect and reflections
                try:
                    from ..memory.unified_storage import update_memory_node_reflections
                    user_reflection = f"User expressed: {user_txt[:100]}..."
                    self_reflection = f"I responded with {final_emotional_state.mood_family if final_emotional_state else 'neutral'} mood: {processed_response[:100]}..."
                    await update_memory_node_reflections(node_id, self_affect, user_reflection, self_reflection)
                    logger.info(f"🎭 Updated memory node {node_id[:8]} with self-affect and reflections")
                except Exception as e:
                    logger.warning(f"🎭 Failed to update memory node {node_id[:8]} with reflections: {e}")
                
                # 🩸 TRIGGER POST-CONVERSATION USER MODEL ANALYSIS
                try:
                    from ..user_modeling.chat_integration import user_modeling_chat_integration
                    
                    # Get current session and emotional state for analysis
                    current_session = CONVERSATION_SESSIONS.get(session_id)
                    if current_session and final_emotional_state:
                        # Create UserModel from emotional processing results
                        current_user_model = emotional_processing_result.get("updated_user_model") if emotional_processing_result else UserModel()
                        
                        # Trigger analysis in background
                        await user_modeling_chat_integration.trigger_post_conversation_analysis(
                            session=current_session,
                            daemon_emotion_state=final_emotional_state,
                            emotional_user_model=current_user_model
                        )
                        
                        logger.info(f"🩸 Triggered post-conversation user model analysis for session {session_id[:8]}")
                
                except Exception as e:
                    logger.warning(f"🩸 Failed to trigger user model analysis: {e}")
                
                # Background processing with the cleaned response
                bg.add_task(process_completed_response_recursion, node_id, processed_response)
                
            except Exception as e:
                logger.debug(f"Streaming paradox processing failed: {e}")
                # Fallback: add unprocessed response to session
                await add_message_to_session(session_id, "assistant", full_response.strip())
                
                # Fallback: Update memory node with basic reflection
                try:
                    from ..memory.unified_storage import update_memory_node_reflections
                    # Classify self affect for fallback case
                    self_affect = await classify_llm_affect(full_response.strip())
                    user_reflection = f"User expressed: {user_txt[:100]}..."
                    self_reflection = f"I responded (fallback): {full_response.strip()[:100]}..."
                    await update_memory_node_reflections(node_id, self_affect, user_reflection, self_reflection)
                    logger.info(f"🎭 Updated memory node {node_id[:8]} with fallback reflections")
                except Exception as reflection_error:
                    logger.warning(f"🎭 Failed to update memory node {node_id[:8]} in fallback: {reflection_error}")
                
                bg.add_task(process_completed_response_recursion, node_id, full_response.strip())

        return StreamingResponse(
            stream_wrapper_for_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming response
        # Get emotional parameters if available
        emotional_params = None
        if emotional_processing_result and emotional_processing_result.get("param_profile"):
            param_profile = emotional_processing_result["param_profile"]
            emotional_params = {
                "target_temperature": param_profile.target_temperature,
                "target_top_p": param_profile.target_top_p,
                "target_max_tokens": getattr(param_profile, 'target_max_tokens', 800)
            }
            logger.info(f"🎭 Applying emotional parameters to non-streaming: temp={param_profile.target_temperature:.3f}")
        
        # Convert to structured messages to preserve conversation history
        if session_messages and len(session_messages) > 1:
            logger.info(f"🎭 Converting to structured messages with {len(session_messages)} conversation messages")
            structured_messages = convert_prompt_to_structured_messages(prompt, session_messages)
            
            # DEBUG: Log the system message to verify daemon personality is preserved
            if structured_messages and structured_messages[0].get("role") == "system":
                system_msg = structured_messages[0]["content"]
                logger.info(f"🎭 System message length: {len(system_msg)} chars")
                logger.debug(f"🎭 System message preview: {system_msg[:300]}...")
                if "daemon" in system_msg.lower():
                    logger.info("🎭 ✓ System message contains daemon personality")
                else:
                    logger.warning("🎭 ✗ System message missing daemon personality!")
            
            # Update response generation status to indicate LLM is now generating
            update_processing_status(session_id, "Response Generation", "processing", "LLM generating response...")
            full_response = await generate_response_for_analysis_with_messages(structured_messages, emotional_params)
        else:
            logger.info(f"🎭 Using traditional prompt (no conversation history)")
            # Update response generation status to indicate LLM is now generating
            update_processing_status(session_id, "Response Generation", "processing", "LLM generating response...")
            full_response = await generate_response_for_analysis(prompt, emotional_params)
        
        # Classify self affect for the response
        self_affect = await classify_llm_affect(full_response)
        
        # 🎭 EMOTIONAL CONSOLIDATION - Complete the emotional processing cycle
        logger.info(f"🎭 EMOTION DEBUG: emotional_processing_result exists: {emotional_processing_result is not None}")
        if emotional_processing_result:
            logger.info(f"🎭 EMOTION DEBUG: Processing result keys: {list(emotional_processing_result.keys()) if emotional_processing_result else 'None'}")
            try:
                # Update the emotional state with the final response and self-affect
                updated_emotion_state = emotional_processing_result["updated_emotion_state"]
                updated_user_model = emotional_processing_result.get("updated_user_model", user_model)
                
                # Store the final response in the episodic trace (the orchestrator already created one)
                # We could enhance this to update the existing trace with the actual response
                logger.info(f"🎭 Emotional consolidation: final response generated with {updated_emotion_state.mood_family} mood")
                
                # CRITICAL FIX: Persist updated emotional state to session storage immediately for dashboard access
                logger.info(f"🎭 EMOTION DEBUG: Session exists: {session is not None}")
                if session:
                    logger.info(f"🎭 EMOTION DEBUG: Session ID: {session.session_id[:8]}, updating emotion state...")
                    session.emotion_state = updated_emotion_state
                    session.user_model = updated_user_model
                    from ..conversations.session_manager import save_session
                    save_session(session)
                    logger.info(f"🎭 EMOTION DEBUG: Session saved successfully")
                    logger.info(f"🎭 Persisted emotional state: {updated_emotion_state.mood_family}, valence={updated_emotion_state.valence:.3f}, arousal={updated_emotion_state.arousal:.3f}")
                else:
                    logger.warning(f"🎭 EMOTION DEBUG: No session found - cannot persist emotion state!")
                
                # DASHBOARD FIX: Store emotional processing results globally for dashboard access
                try:
                    from ..config import DASHBOARD_STATE_CACHE
                    logger.info(f"🎭 EMOTION DEBUG: DASHBOARD_STATE_CACHE exists: {DASHBOARD_STATE_CACHE is not None}")
                    DASHBOARD_STATE_CACHE.update({
                        "current_emotion_state": updated_emotion_state,
                        "current_user_model": updated_user_model,
                        "active_seeds": emotional_processing_result.get("applied_seeds", []),
                        "distortion_frame": emotional_processing_result.get("distortion_frame"),
                        "reasoning_steps": emotional_processing_result.get("reasoning_steps", {}),
                        "last_updated": datetime.now(timezone.utc),
                        "session_id": session_id
                    })
                    logger.info(f"🎭 EMOTION DEBUG: Dashboard cache updated with {len(DASHBOARD_STATE_CACHE)} items")
                    logger.info(f"🎭 EMOTION DEBUG: Cache keys: {list(DASHBOARD_STATE_CACHE.keys())}")
                except Exception as cache_error:
                    logger.error(f"🎭 Dashboard cache update failed: {cache_error}")
                    import traceback
                    logger.error(f"🎭 Cache error traceback: {traceback.format_exc()}")
                
            except Exception as e:
                logger.error(f"🎭 EMOTION DEBUG: Emotional consolidation failed: {e}")
                import traceback
                logger.error(f"🎭 EMOTION DEBUG: Consolidation error traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"🎭 EMOTION DEBUG: No emotional_processing_result - skipping consolidation")
        
        # Update turn analysis with self affect
        turn_analyzer.update_emotion_analysis(session_id, turn_id, {
            "user_affect_magnitude": sum(abs(x) for x in user_affect),
            "self_affect_magnitude": sum(abs(x) for x in self_affect),
            "dominant_user_emotions": [
                label for label, score in zip(["joy", "sadness", "anger", "fear", "surprise", "disgust"], user_affect[:6])
                if score > 0.5
            ][:3],
            "dominant_self_emotions": [
                label for label, score in zip(["joy", "sadness", "anger", "fear", "surprise", "disgust"], self_affect[:6])
                if score > 0.5
            ][:3],
            "emotional_influence_score": max(user_affect + self_affect) if (user_affect + self_affect) else 0.0
        })
        
        # Add assistant response to session
        await add_message_to_session(session_id, "assistant", full_response, self_affect=self_affect)
        
        # Store episodic trace for this completed turn
        try:
            from ..emotions.memory_store import create_and_store_episodic_trace
            final_emotional_state = emotional_processing_result["updated_emotion_state"] if emotional_processing_result else None
            await create_and_store_episodic_trace(
                session_id=session_id,
                turn_id=turn_number,
                user_input=user_txt,
                ai_response=full_response,
                user_affect=user_affect,
                self_affect=self_affect,
                emotional_state=final_emotional_state,
                context_synopses=ctx_syn,
                reflection=f"Non-streaming turn completed"
            )
            logger.info(f"🎭 Stored episodic trace for non-streaming turn {turn_number}")
        except Exception as e:
            logger.warning(f"🎭 Failed to store episodic trace for non-streaming turn: {e}")
        
        # Update the initially stored dual-affect memory node with self-affect and reflections
        try:
            from ..memory.unified_storage import update_memory_node_reflections
            user_reflection = f"User expressed: {user_txt[:100]}..."
            self_reflection = f"I responded with {final_emotional_state.mood_family if final_emotional_state else 'neutral'} mood: {full_response[:100]}..."
            await update_memory_node_reflections(node_id, self_affect, user_reflection, self_reflection)
            logger.info(f"🎭 Updated memory node {node_id[:8]} with self-affect and reflections")
        except Exception as e:
            logger.warning(f"🎭 Failed to update memory node {node_id[:8]} with reflections: {e}")
        
        # 🩸 TRIGGER POST-CONVERSATION USER MODEL ANALYSIS
        try:
            from ..user_modeling.chat_integration import user_modeling_chat_integration
            
            # Get current session and emotional state for analysis
            current_session = CONVERSATION_SESSIONS.get(session_id)
            if current_session and final_emotional_state:
                # Create UserModel from emotional processing results
                current_user_model = emotional_processing_result.get("updated_user_model") if emotional_processing_result else UserModel()
                
                # Trigger analysis in background
                await user_modeling_chat_integration.trigger_post_conversation_analysis(
                    session=current_session,
                    daemon_emotion_state=final_emotional_state,
                    emotional_user_model=current_user_model
                )
                
                logger.info(f"🩸 Triggered post-conversation user model analysis for session {session_id[:8]}")
        
        except Exception as e:
            logger.warning(f"🩸 Failed to trigger user model analysis: {e}")
        
        # PARADOX SYSTEM INTEGRATION - Process response for paradoxes
        paradox_results = {}
        try:
            from ..paradox.integration import process_ai_response_with_paradox
            
            # Calculate affect delta for paradox detection
            affect_delta = sum(abs(u - s) for u, s in zip(user_affect, self_affect)) / len(user_affect)
            
            # Process response through paradox system
            paradox_results = await process_ai_response_with_paradox(
                user_message=user_txt,
                ai_response=full_response,
                context_memories=[{"id": f"ctx_{i}", "text": ctx} for i, ctx in enumerate(ctx_syn)] if ctx_syn else [],
                emotion_state={"user_affect": user_affect, "ai_affect": self_affect},
                affect_delta=affect_delta
            )
            
            if paradox_results.get('paradox_detected'):
                logger.info(f"🌪️ Paradox detected: {paradox_results['paradox_data']['paradox_type']}")
                
                # Use cleaned response if available
                if paradox_results.get('cleaned_response'):
                    full_response = paradox_results['cleaned_response']
            
        except Exception as e:
            logger.debug(f"Paradox processing failed: {e}")
        
        # Add background task for non-streaming response processing
        bg.add_task(process_completed_response_recursion, node_id, full_response)
        
        # Calculate context usage for dashboard
        session = CONVERSATION_SESSIONS.get(session_id)
        context_tokens = len(ctx_syn) * 10 if ctx_syn else 0  # Rough estimate
        prompt_tokens = estimate_token_count(prompt)
        completion_tokens = estimate_token_count(full_response)
        total_tokens = prompt_tokens + completion_tokens
        
        # Update turn analysis with processing stats
        turn_analyzer.update_processing_stats(session_id, turn_id, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "generation_time_ms": 0  # Would need to measure this
        })
        
        # Complete response generation
        update_processing_status(session_id, "Response Generation", "completed", "Response generation complete")
        
        # Don't clear processing status immediately - let the auto-clear handle it after a delay
        
        # Complete the turn analysis
        turn_analyzer.complete_turn_analysis(session_id, turn_id, full_response)
        
        return {
            "model": req.model,
            "choices": [{"message": {"role": "assistant", "content": full_response}}],
            "usage": {
                "prompt_tokens": prompt_tokens, 
                "completion_tokens": completion_tokens, 
                "total_tokens": total_tokens
            },
            "session_id": session_id,
            "turn_id": turn_id,
            "context_usage": {
                "context_tokens": context_tokens,
                "total_session_tokens": session.total_tokens if session else total_tokens,
                "message_count": len(session.messages) if session else 2
            },
            "debug_info": {
                "memory_stored": True,
                "affect_processing": "dual_channel",
                "context_retrieved": len(ctx_syn) if ctx_syn else 0,
                "turn_analysis_available": True
            }
        }

@router.get("/v1/processing/status", summary="Get current processing status")
async def get_current_processing_status():
    """Get current processing status for active session"""
    try:
        from ..conversations import get_or_create_active_session
        
        # Get active session
        session_id = await get_or_create_active_session()
        if not session_id:
            return {"processing_steps": [], "is_processing": False}
        processing_status = get_processing_status(session_id)
        
        # Convert to frontend format
        steps = []
        step_names = ["Input Analysis", "Emotional Processing", "Memory Retrieval", "Thinking Layer", "Response Generation"]
        
        all_completed = True
        any_processing = False
        
        for step_name in step_names:
            if step_name in processing_status:
                step_data = processing_status[step_name]
                steps.append({
                    "name": step_name,
                    "description": step_data["description"],
                    "status": step_data["status"],
                    "timestamp": step_data["timestamp"]
                })
                
                if step_data["status"] == "processing":
                    any_processing = True
                    all_completed = False
                elif step_data["status"] != "completed":
                    all_completed = False
            else:
                steps.append({
                    "name": step_name,
                    "description": f"{step_name} pending...",
                    "status": "pending"
                })
                all_completed = False
        
        # Check if all steps are completed and enough time has passed
        if all_completed and processing_status:
            # Check if 3 seconds have passed since last update
            import time
            current_time = time.time()
            last_update_time = 0
            
            for step_data in processing_status.values():
                try:
                    step_time = datetime.fromisoformat(step_data["timestamp"].replace('Z', '+00:00')).timestamp()
                    last_update_time = max(last_update_time, step_time)
                except:
                    pass
            
            # Auto-clear after 3 seconds to stop showing completed steps
            if current_time - last_update_time > 3:
                force_clear_processing_status(session_id)
                return {"processing_steps": [], "is_processing": False, "session_id": session_id}
        
        is_processing = any_processing or (len(processing_status) > 0 and not all_completed)
        
        return {
            "processing_steps": steps,
            "is_processing": is_processing,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return {"processing_steps": [], "is_processing": False, "error": str(e)}

@router.post("/v1/processing/clear", summary="Clear processing status")
async def clear_processing_status_endpoint():
    """Clear all processing status (useful when switching conversations)"""
    try:
        clear_all_processing_status()
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing processing status: {e}")
        return {"error": str(e)}