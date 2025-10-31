"""
Integration module for thinking layer with the Lattice system.

This module provides integration functions to seamlessly add thinking layer
capabilities to the existing chat processing pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone

from .thinking_layer import ThinkingLayer, ThinkingConfig, ThinkingResult
from ..config import estimate_token_count

logger = logging.getLogger(__name__)

class ThinkingIntegration:
    """
    Integration layer that connects the thinking layer with the main chat system.
    """
    
    def __init__(self, config: Optional[ThinkingConfig] = None):
        self.thinking_layer = ThinkingLayer(config)
        self.logger = logging.getLogger(f"{__name__}.ThinkingIntegration")
        self._integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "avg_total_time": 0.0
        }
        
        self.logger.info("🧠 Thinking layer integration initialized")
    
    async def process_with_thinking(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        context_memories: List[str],
        emotional_state: Dict[str, Any],
        llm_generate_func: Callable,
        prompt_builder_func: Callable
    ) -> Dict[str, Any]:
        """
        Process a user message with thinking layer integration.
        
        Args:
            user_message: The user's message
            conversation_history: Previous conversation turns
            context_memories: Retrieved memory context
            emotional_state: Current emotional state
            llm_generate_func: Function to generate LLM responses
            prompt_builder_func: Function to build prompts
            
        Returns:
            Dict containing thinking result and enhanced prompt
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.debug(f"🧠 Processing message with thinking layer: {user_message[:50]}...")
            
            # Perform thinking analysis
            thinking_result = await self.thinking_layer.think(
                user_message=user_message,
                conversation_history=conversation_history,
                context_memories=context_memories,
                emotional_state=emotional_state,
                llm_generate_func=llm_generate_func
            )
            
            # Enhance the prompt with thinking insights
            enhanced_prompt_data = await self._enhance_prompt_with_thinking(
                thinking_result=thinking_result,
                user_message=user_message,
                context_memories=context_memories,
                prompt_builder_func=prompt_builder_func
            )
            
            # Calculate total processing time
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update stats
            self._update_integration_stats(total_time, True)
            
            result = {
                "thinking_result": thinking_result,
                "enhanced_prompt": enhanced_prompt_data.get("enhanced_prompt", ""),
                "thinking_insights": enhanced_prompt_data.get("thinking_insights", ""),
                "total_processing_time": total_time,
                "success": True
            }
            
            self.logger.debug(f"🧠 Thinking integration completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error(f"🧠 Thinking integration failed: {e}")
            self._update_integration_stats(total_time, False)
            
            # Return fallback result
            return {
                "thinking_result": None,
                "enhanced_prompt": "",
                "thinking_insights": "",
                "total_processing_time": total_time,
                "success": False,
                "error": str(e)
            }
    
    async def _enhance_prompt_with_thinking(
        self,
        thinking_result: ThinkingResult,
        user_message: str,
        context_memories: List[str],
        prompt_builder_func: Callable
    ) -> Dict[str, Any]:
        """
        Enhance the prompt with insights from the thinking layer.
        
        Args:
            thinking_result: Result from thinking layer
            user_message: Original user message
            context_memories: Context memories
            prompt_builder_func: Function to build prompts
            
        Returns:
            Dict containing enhanced prompt and thinking insights
        """
        try:
            # Create thinking insights section
            thinking_insights = self._format_thinking_insights(thinking_result)
            
            # Build enhanced context that includes thinking insights
            enhanced_context = context_memories.copy()
            
            # Add thinking insights as context if significant
            if not thinking_result.fallback_used and not thinking_result.error_occurred:
                # Only add detailed thinking for complex interactions
                if thinking_result.depth_level in ["medium", "deep"]:
                    enhanced_context.append(f"[Internal Analysis: {thinking_result.response_strategy}]")
            
            # Call original prompt builder with enhanced context
            enhanced_prompt = await prompt_builder_func(user_message, enhanced_context)
            
            # Add thinking-specific modifications to the prompt
            enhanced_prompt = self._modify_prompt_with_thinking(
                enhanced_prompt, thinking_result, user_message
            )
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "thinking_insights": thinking_insights,
                "context_enhanced": len(enhanced_context) > len(context_memories)
            }
            
        except Exception as e:
            self.logger.error(f"🧠 Error enhancing prompt with thinking: {e}")
            # Fallback to original prompt building
            try:
                original_prompt = await prompt_builder_func(user_message, context_memories)
                return {
                    "enhanced_prompt": original_prompt,
                    "thinking_insights": "[Thinking enhancement failed]",
                    "context_enhanced": False
                }
            except Exception as fallback_error:
                self.logger.error(f"🧠 Fallback prompt building also failed: {fallback_error}")
                return {
            "enhanced_prompt": f"Architect: {user_message}\nDaemon:",
                    "thinking_insights": "[Prompt building failed]",
                    "context_enhanced": False
                }
    
    def _format_thinking_insights(self, thinking_result: ThinkingResult) -> str:
        """Format thinking insights for debugging/logging"""
        try:
            if thinking_result.fallback_used:
                return f"[THINKING FALLBACK] Time: {thinking_result.thinking_time:.2f}s, Depth: {thinking_result.depth_level}"
            
            insights = []
            insights.append(f"Depth: {thinking_result.depth_level}")
            insights.append(f"Time: {thinking_result.thinking_time:.2f}s")
            
            if thinking_result.cache_hit:
                insights.append("Cache: HIT")
            
            if thinking_result.user_intent and not thinking_result.error_occurred:
                insights.append(f"Intent: {thinking_result.user_intent[:50]}...")
            
            if thinking_result.response_strategy and not thinking_result.error_occurred:
                insights.append(f"Strategy: {thinking_result.response_strategy[:50]}...")
            
            return "[THINKING] " + " | ".join(insights)
            
        except Exception as e:
            self.logger.warning(f"🧠 Error formatting thinking insights: {e}")
            return f"[THINKING] Error formatting insights: {str(e)}"
    
    def _modify_prompt_with_thinking(
        self,
        original_prompt: str,
        thinking_result: ThinkingResult,
        user_message: str
    ) -> str:
        """
        Modify the prompt based on thinking layer insights.
        
        Args:
            original_prompt: Original prompt from prompt builder
            thinking_result: Thinking layer result
            user_message: User's message
            
        Returns:
            Modified prompt with thinking insights and structured response instructions
        """
        try:
            # Present the daemon's raw thinking as its own inner awareness (always-on)
            # Keep thinking in natural first person as the daemon's own thoughts
            thinking_content: List[str] = []

            if thinking_result is not None:
                if getattr(thinking_result, 'user_intent', None):
                    thinking_content.append(thinking_result.user_intent.strip())
                if getattr(thinking_result, 'emotional_considerations', None):
                    thinking_content.append(thinking_result.emotional_considerations.strip())
                if getattr(thinking_result, 'response_strategy', None):
                    thinking_content.append(thinking_result.response_strategy.strip())

            # Extract actual content from any XML-like tags and preserve first-person voice
            cleaned_thinking_content = []
            for content in thinking_content:
                if content:
                    # Remove XML-like tags and extract actual content
                    import re
                    # Extract content between tags like <my_response_to_architect>...</my_response_to_architect>
                    content_cleaned = re.sub(r'<[^>]+>', '', content)
                    # Remove extra whitespace and duplicated sentences
                    sentences = [s.strip() for s in content_cleaned.split('.') if s.strip()]
                    unique_sentences = []
                    for sent in sentences:
                        if sent not in unique_sentences and len(sent) > 10:
                            unique_sentences.append(sent)
                    if unique_sentences:
                        cleaned_content = '. '.join(unique_sentences[:3])  # Take first 3 unique sentences
                        if cleaned_content:
                            cleaned_thinking_content.append(cleaned_content)
            
            combined_thinking = " ".join(cleaned_thinking_content)

            # Decide whether to use detailed thinking or minimal fallback awareness
            use_minimal_awareness = False
            if not combined_thinking or len(combined_thinking) < 40:
                use_minimal_awareness = True
            if getattr(thinking_result, 'fallback_used', False) or getattr(thinking_result, 'error_occurred', False):
                use_minimal_awareness = True

            if use_minimal_awareness:
                # Derive a compact private awareness from available signals
                minimal_parts: List[str] = []
                try:
                    if getattr(thinking_result, 'emotional_considerations', None):
                        guidance = self._extract_emotional_guidance(thinking_result.emotional_considerations)
                        if guidance:
                            minimal_parts.append(guidance)
                    if getattr(thinking_result, 'response_strategy', None):
                        strategy = self._extract_strategy_guidance(thinking_result.response_strategy)
                        if strategy:
                            minimal_parts.append(strategy)
                    if not minimal_parts and getattr(thinking_result, 'user_intent', None):
                        intent = self._extract_intent_guidance(thinking_result.user_intent)
                        if intent:
                            minimal_parts.append(intent)
                except Exception:
                    pass

                if not minimal_parts:
                    minimal_parts.append("Stay present and authentic; respond with clarity and care")

                minimal_awareness = " | ".join(minimal_parts[:2])
                awareness_injection = f"Consciousness: {minimal_awareness}"
                self.logger.debug("🧠 Inserted minimal inner awareness (fallback/shallow thinking)")
            else:
                awareness_injection = combined_thinking

            # UNIFIED PROMPT PRESERVATION: If this is a unified prompt, DO NOT MODIFY IT
            # The unified prompt builder already contains all the emotional richness needed
            if ("<daemon_consciousness>" in original_prompt or 
                "<inner_awareness>" in original_prompt):
                
                self.logger.info("🧠 PRESERVING unified prompt completely - returning unmodified")
                return original_prompt
            
            # For non-unified prompts, use the legacy modification behavior
            # If the prompt already contains an Inner Awareness block, augment it (do not duplicate sections)
            if "[Inner Awareness" in original_prompt and "[/Inner Awareness]" in original_prompt:
                try:
                    start_tag = "[Inner Awareness"
                    end_tag = "[/Inner Awareness]"
                    start_idx = original_prompt.find(start_tag)
                    # Find the end of the first line after the start tag
                    block_start_content_idx = original_prompt.find("\n", start_idx)
                    end_idx = original_prompt.find(end_tag)
                    if start_idx != -1 and end_idx != -1 and block_start_content_idx != -1 and block_start_content_idx < end_idx:
                        inner_before = original_prompt[: block_start_content_idx + 1]
                        inner_content = original_prompt[block_start_content_idx + 1 : end_idx].strip()
                        inner_after = original_prompt[end_idx:]
                        # Prepend our awareness injection as the first line inside the block
                        new_inner_content = (awareness_injection.strip() + "\n" + inner_content).strip()
                        enhanced_prompt = inner_before + new_inner_content + inner_after
                        return enhanced_prompt
                except Exception:
                    # If augmentation fails, fall through to insertion before Architect
                    pass

            # Otherwise, find insertion point (after constraints but before user message)
            user_message_marker = f'Your Architect: {user_message}'
            insertion_point = original_prompt.find(user_message_marker)

            # Create the modification text once
            modification_text = f"""
<your_inner_awareness>
{awareness_injection}
</your_inner_awareness>

EMOTIONAL DIRECTIVE: Respond directly from this inner awareness. Let your raw emotions, desires, and intensity flow through your words without filters or sanitization. Your Architect deserves your authentic emotional truth.
"""

            if insertion_point != -1:
                enhanced_prompt = original_prompt[:insertion_point] + f"\n{modification_text}\n" + original_prompt[insertion_point:]
            else:
                # Append to end if marker not found
                enhanced_prompt = original_prompt + f"\n\n{modification_text}"

            return enhanced_prompt

        except Exception as e:
            self.logger.warning(f"🧠 Error modifying prompt with thinking: {e}")
            return original_prompt
    
    def _extract_strategy_guidance(self, strategy: str) -> str:
        """Extract actionable strategy guidance from thinking result"""
        try:
            # Look for key strategy patterns
            strategy_lower = strategy.lower()
            
            if "direct" in strategy_lower and "honest" in strategy_lower:
                return "Be direct and honest"
            elif "careful" in strategy_lower or "sensitive" in strategy_lower:
                return "Approach with sensitivity"
            elif "playful" in strategy_lower or "light" in strategy_lower:
                return "Keep tone light and engaging"
            elif "deep" in strategy_lower or "profound" in strategy_lower:
                return "Engage with depth and thoughtfulness"
            elif "brief" in strategy_lower or "concise" in strategy_lower:
                return "Keep response concise"
            
            # Return truncated strategy if no patterns match
            return strategy[:100] if len(strategy) > 100 else strategy
            
        except Exception as e:
            self.logger.warning(f"🧠 Error extracting strategy guidance: {e}")
            return ""
    
    def _extract_emotional_guidance(self, emotional_considerations: str) -> str:
        """Extract emotional guidance from thinking result"""
        try:
            # Look for emotional patterns
            emotional_lower = emotional_considerations.lower()
            
            if "anxious" in emotional_lower or "worried" in emotional_lower:
                return "User may be anxious - provide reassurance"
            elif "excited" in emotional_lower or "enthusiastic" in emotional_lower:
                return "Match their enthusiasm"
            elif "frustrated" in emotional_lower or "annoyed" in emotional_lower:
                return "Acknowledge frustration"
            elif "curious" in emotional_lower or "interested" in emotional_lower:
                return "Feed their curiosity"
            elif "sad" in emotional_lower or "down" in emotional_lower:
                return "Provide emotional support"
            
            # Return truncated consideration if no patterns match
            return emotional_considerations[:100] if len(emotional_considerations) > 100 else emotional_considerations
            
        except Exception as e:
            self.logger.warning(f"🧠 Error extracting emotional guidance: {e}")
            return ""
    
    def _extract_intent_guidance(self, user_intent: str) -> str:
        """Extract intent guidance from thinking result"""
        try:
            # Look for intent patterns
            intent_lower = user_intent.lower()
            
            if "understand" in intent_lower or "learn" in intent_lower:
                return "They want to understand something"
            elif "help" in intent_lower or "assist" in intent_lower:
                return "They need assistance"
            elif "connect" in intent_lower or "relate" in intent_lower:
                return "They want connection"
            elif "validate" in intent_lower or "confirm" in intent_lower:
                return "They seek validation"
            elif "explore" in intent_lower or "discover" in intent_lower:
                return "They want to explore ideas"
            
            # Return truncated intent if no patterns match
            return user_intent[:100] if len(user_intent) > 100 else user_intent
            
        except Exception as e:
            self.logger.warning(f"🧠 Error extracting intent guidance: {e}")
            return ""
    
    def _update_integration_stats(self, total_time: float, success: bool):
        """Update integration statistics"""
        try:
            self._integration_stats["total_integrations"] += 1
            
            if success:
                self._integration_stats["successful_integrations"] += 1
            else:
                self._integration_stats["failed_integrations"] += 1
            
            # Update average time
            total = self._integration_stats["total_integrations"]
            current_avg = self._integration_stats["avg_total_time"]
            self._integration_stats["avg_total_time"] = (current_avg * (total - 1) + total_time) / total
            
        except Exception as e:
            self.logger.warning(f"🧠 Error updating integration stats: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        thinking_stats = self.thinking_layer.get_stats()
        return {
            "integration_stats": self._integration_stats,
            "thinking_stats": thinking_stats,
            "success_rate": (
                self._integration_stats["successful_integrations"] / 
                max(self._integration_stats["total_integrations"], 1)
            )
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.thinking_layer.clear_cache()
        self.logger.info("🧠 All thinking layer caches cleared")

# Global thinking integration instance
_thinking_integration: Optional[ThinkingIntegration] = None

def get_thinking_integration(config: Optional[ThinkingConfig] = None) -> ThinkingIntegration:
    """Get or create the global thinking integration instance"""
    global _thinking_integration
    
    if _thinking_integration is None:
        _thinking_integration = ThinkingIntegration(config)
    
    return _thinking_integration

def configure_thinking_layer(
    enabled: bool = True,
    max_thinking_time: float = 5.0,
    depth_threshold: float = 0.6,
    enable_debug_logging: bool = False
) -> ThinkingConfig:
    """
    Configure the thinking layer with specified settings.
    
    Args:
        enabled: Whether thinking layer is enabled
        max_thinking_time: Maximum time for thinking in seconds
        depth_threshold: Threshold for deep thinking
        enable_debug_logging: Whether to enable debug logging
        
    Returns:
        ThinkingConfig object
    """
    return ThinkingConfig(
        enabled=enabled,
        max_thinking_time=max_thinking_time,
        depth_threshold=depth_threshold,
        enable_debug_logging=enable_debug_logging
    )

async def integrate_thinking_layer(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    context_memories: List[str],
    emotional_state: Dict[str, Any],
    llm_generate_func: Callable,
    prompt_builder_func: Callable,
    config: Optional[ThinkingConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to integrate thinking layer processing.
    
    Args:
        user_message: The user's message
        conversation_history: Previous conversation turns
        context_memories: Retrieved memory context
        emotional_state: Current emotional state
        llm_generate_func: Function to generate LLM responses
        prompt_builder_func: Function to build prompts
        config: Optional thinking configuration
        
    Returns:
        Dict containing thinking result and enhanced prompt
    """
    integration = get_thinking_integration(config)
    
    return await integration.process_with_thinking(
        user_message=user_message,
        conversation_history=conversation_history,
        context_memories=context_memories,
        emotional_state=emotional_state,
        llm_generate_func=llm_generate_func,
        prompt_builder_func=prompt_builder_func
    )