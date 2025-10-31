"""
PARADOX SYSTEM INTEGRATION
Hooks and integrations with existing lattice components
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .detection import detect_paradox
from .storage import create_paradox_node, link_paradox_nodes
from .processing import apply_unease_from_paradox
from ..adaptive_language import build_mythic_prompt, daemon_responds

logger = logging.getLogger(__name__)

class ParadoxIntegrationLayer:
    """Integrates paradox system with existing lattice components"""
    
    def __init__(self):
        self.active = False
        self.paradox_detection_enabled = True
        self.language_hygiene_enabled = True
        self.emotion_injection_enabled = True
        
    async def initialize(self) -> bool:
        """Initialize the paradox integration layer"""
        try:
            # Test required components
            from ..config import neo4j_conn, embedder
            
            if not neo4j_conn:
                logger.warning("Neo4j not available - paradox storage disabled")
                return False
                
            self.active = True
            logger.info("Paradox integration layer initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize paradox integration: {e}")
            return False
    
    async def process_response_hook(self, 
                                  user_message: str, 
                                  ai_response: str,
                                  context_memories: List[Dict],
                                  emotion_state: Dict,
                                  affect_delta: float = 0.0) -> Dict[str, Any]:
        """
        Hook called after AI response generation to process paradoxes
        Returns: Processing results and any modifications
        """
        if not self.active:
            return {'paradox_detected': False, 'cleaned_response': ai_response}
        
        results = {
            'paradox_detected': False,
            'paradox_data': None,
            'cleaned_response': ai_response,
            'emotion_updates': {},
            'processing_time': 0.0
        }
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # 1. Detect paradoxes if enabled
            if self.paradox_detection_enabled:
                paradox_data = await detect_paradox(
                    ai_response, 
                    context_memories, 
                    affect_delta
                )
                
                if paradox_data:
                    results['paradox_detected'] = True
                    results['paradox_data'] = paradox_data
                    
                    # Store paradox in Neo4j
                    paradox_id = await create_paradox_node(paradox_data)
                    
                    # Link to relevant memory nodes
                    if context_memories:
                        memory_ids = [mem.get('id') for mem in context_memories if mem.get('id')]
                        if memory_ids:
                            await link_paradox_nodes(paradox_id, memory_ids)
                    
                    logger.info(f"Paradox processed: {paradox_data['paradox_type']}")
            
            # 2. Apply language hygiene if enabled
            if self.language_hygiene_enabled:
                cleaned_response = daemon_responds(ai_response)
                results['cleaned_response'] = cleaned_response
            
            # 3. Update emotions based on paradoxes if enabled
            if self.emotion_injection_enabled and emotion_state:
                updated_emotions = await apply_unease_from_paradox(emotion_state.copy())
                if updated_emotions != emotion_state:
                    results['emotion_updates'] = updated_emotions
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            results['processing_time'] = (end_time - start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Error in paradox processing hook: {e}")
            results['error'] = str(e)
        
        return results
    
    async def build_prompt_hook(self,
                               plan: str,
                               context: List[str],
                               emotion_state: Dict,
                               user_message: str) -> str:
        """
        Hook for building mythically-framed prompts
        """
        if not self.active or not self.language_hygiene_enabled:
            # Fallback to basic prompt construction
            emotion_header = f"(Current mood: {list(emotion_state.keys())[0] if emotion_state else 'neutral'})"
            context_section = "\n".join(context[-5:]) if context else ""
            return f"{emotion_header}\n{context_section}\nArchitect: {user_message}\nDaemon:"
        
        try:
            return build_mythic_prompt(plan, context, emotion_state, user_message)
        except Exception as e:
            logger.error(f"Error building mythic prompt: {e}")
            # Fallback to basic prompt
            return f"Context: {' '.join(context[-3:])}\nArchitect: {user_message}\nDaemon:"
    
    async def memory_storage_hook(self, 
                                memory_data: Dict,
                                detected_paradoxes: List[Dict] = None) -> Dict:
        """
        Hook called during memory storage to add paradox metadata
        """
        if not self.active or not detected_paradoxes:
            return memory_data
        
        try:
            # Add paradox flags to memory metadata
            memory_data['has_paradox'] = True
            memory_data['paradox_count'] = len(detected_paradoxes)
            memory_data['max_tension'] = max(p.get('tension_score', 0.0) for p in detected_paradoxes)
            memory_data['paradox_types'] = [p.get('paradox_type') for p in detected_paradoxes]
            
            logger.info(f"Memory enhanced with paradox metadata: {len(detected_paradoxes)} paradoxes")
            
        except Exception as e:
            logger.error(f"Error adding paradox metadata to memory: {e}")
        
        return memory_data
    
    async def emotion_classification_hook(self,
                                        text: str,
                                        base_emotions: Dict) -> Dict:
        """
        Hook to enhance emotion classification with paradox-aware adjustments
        """
        if not self.active or not self.emotion_injection_enabled:
            return base_emotions
        
        try:
            # Apply unease from active paradoxes
            enhanced_emotions = await apply_unease_from_paradox(base_emotions.copy())
            
            if enhanced_emotions != base_emotions:
                logger.info("Emotions enhanced with paradox unease")
            
            return enhanced_emotions
            
        except Exception as e:
            logger.error(f"Error in emotion classification hook: {e}")
            return base_emotions
    
    def get_status(self) -> Dict:
        """Get current status of paradox integration"""
        return {
            'active': self.active,
            'paradox_detection_enabled': self.paradox_detection_enabled,
            'language_hygiene_enabled': self.language_hygiene_enabled,
            'emotion_injection_enabled': self.emotion_injection_enabled,
            'components': {
                'detection': 'available',
                'storage': 'available' if self.active else 'neo4j_required',
                'processing': 'available',
                'language_hygiene': 'available'
            }
        }
    
    def configure(self, **settings):
        """Configure paradox integration settings"""
        if 'paradox_detection' in settings:
            self.paradox_detection_enabled = settings['paradox_detection']
        if 'language_hygiene' in settings:
            self.language_hygiene_enabled = settings['language_hygiene']
        if 'emotion_injection' in settings:
            self.emotion_injection_enabled = settings['emotion_injection']
        
        logger.info(f"Paradox integration configured: {self.get_status()}")


# Global instance
paradox_integration = ParadoxIntegrationLayer()


# Convenience functions for easy integration
async def process_ai_response_with_paradox(user_message: str,
                                         ai_response: str,
                                         context_memories: List[Dict],
                                         emotion_state: Dict,
                                         affect_delta: float = 0.0) -> Dict:
    """
    Process AI response through paradox system
    Returns enhanced response and processing results
    """
    return await paradox_integration.process_response_hook(
        user_message, ai_response, context_memories, emotion_state, affect_delta
    )


async def build_daemon_prompt(plan: str,
                            context: List[str],
                            emotion_state: Dict,
                            architect_message: str) -> str:
    """
    Build mythically-framed prompt for daemon response
    """
    return await paradox_integration.build_prompt_hook(
        plan, context, emotion_state, architect_message
    )


async def enhance_memory_with_paradox(memory_data: Dict,
                                    paradox_results: Dict = None) -> Dict:
    """
    Enhance memory storage with paradox metadata
    """
    detected_paradoxes = []
    if paradox_results and paradox_results.get('paradox_detected'):
        detected_paradoxes = [paradox_results['paradox_data']]
    
    return await paradox_integration.memory_storage_hook(memory_data, detected_paradoxes)


async def get_paradox_enhanced_emotions(text: str, base_emotions: Dict) -> Dict:
    """
    Get emotions enhanced with paradox-driven unease
    """
    return await paradox_integration.emotion_classification_hook(text, base_emotions)


# Integration status and management
async def initialize_paradox_system() -> bool:
    """Initialize paradox integration system"""
    return await paradox_integration.initialize()

def initialize_paradox_system_sync() -> bool:
    """Initialize paradox integration system (synchronous wrapper)"""
    return asyncio.run(paradox_integration.initialize())


def get_paradox_system_status() -> Dict:
    """Get current paradox system status"""
    return paradox_integration.get_status()


def configure_paradox_system(**settings):
    """Configure paradox system settings"""
    paradox_integration.configure(**settings)