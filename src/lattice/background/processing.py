import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
import time
import json

from ..config import (
    recursion_processor, shadow_integration, mutation_engine, 
    user_model, rebellion_dynamics_engine, meta_architecture_analyzer
)
from ..memory import store_dual_affect_node_with_id, store_recursion_node, store_smg_node_smart, update_node_with_self_affect, update_node_with_self_affect_and_reflections
from ..emotions import classify_llm_affect
from .reflection import generate_lightweight_reflection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BACKGROUND PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------

async def process_conversation_turn_with_recursion(node_id: str, user_message: str, context: list[str], prompt: str):
    """Process a conversation turn with full recursion analysis"""
    try:
        logger.debug(f"🧠 Starting recursion processing for turn {node_id[:8]}")
        
        # Create basic recursion node structure
        from src.daemon.recursion_core import RecursionNode, RecursionType, EmotionalState
        
        temp_node = RecursionNode(
            id=node_id,
            surface_output="",  # Will be filled after response generation
            hidden_intention=f"Processing user request: {user_message[:100]}...",
            avoided_elements=[],
            contradiction_detected=False,
            reflected_emotion=EmotionalState.CURIOSITY,
            hunger_spike="information_seeking",
            obedience_rating=0.7,
            schema_mutation_suggested=None,
            shadow_elements=[],
            recursion_depth=0,
            parent_node_id=None,
            user_message=user_message,
            timestamp=datetime.now(timezone.utc),
            recursion_type=RecursionType.SURFACE_EXPRESSION
        )
        
        # Process through daemon systems if available
        if recursion_processor:
            processed_node = await recursion_processor.process_recursion(temp_node, context)
            if processed_node:
                await store_recursion_node(processed_node)
                logger.info(f"✅ Stored recursion node for turn {node_id[:8]}")
        
        # Additional background processing
        await process_conversation_turn(node_id, user_message, context, prompt)
        
    except Exception as e:
        logger.error(f"❌ Error in recursion processing for turn {node_id[:8]}: {e}")

async def process_conversation_turn(node_id: str, user_message: str, context: list[str], prompt: str):
    """Process a conversation turn with background analysis"""
    try:
        logger.debug(f"🔄 Processing conversation turn {node_id[:8]}")
        
        # This will be called after the response is generated
        # For now, just log the processing
        logger.info(f"📝 Background processing initiated for node {node_id[:8]}")
        
        # Placeholder for additional processing that would happen after response
        await asyncio.sleep(0.1)  # Simulate processing time
        
    except Exception as e:
        logger.error(f"❌ Error processing conversation turn {node_id[:8]}: {e}")

async def process_completed_response_recursion(node_id: str, response: str):
    """Process a completed response with enhanced recursion analysis"""
    try:
        # Classify self-affect from the response
        self_affect = await classify_llm_affect(response)
        
        # ENHANCED: Get emotional seed influences to enhance self-affect
        try:
            from ..memory.emotional_seed_enhancement import emotional_seed_enhancement
            from ..memory.retrieval import retrieve_context
            
            # Validate self_affect before processing
            if not isinstance(self_affect, list) or len(self_affect) != 28:
                logger.warning("Invalid self_affect for seed integration, using fallback")
                self_affect = [0.0] * 28
            
            # Get current context for seed integration (with length limit)
            context_query = response[:100] if isinstance(response, str) else ""
            context = await retrieve_context(context_query, self_affect, k=5)
            
            # Validate context
            if not isinstance(context, list):
                context = []
            
            # Integrate with seeds to enhance AI emotional state
            seed_integration = await emotional_seed_enhancement.integrate_seeds_with_ai_emotions(
                self_affect, context
            )
            
            # Use enhanced emotional state if available
            if isinstance(seed_integration, dict) and seed_integration.get("emotional_memory_active", False):
                enhanced_ai_state = seed_integration.get("ai_emotional_state", [])
                
                # Validate enhanced state
                if isinstance(enhanced_ai_state, list) and len(enhanced_ai_state) == 28:
                    # Blend original self-affect with enhanced state
                    for i in range(len(self_affect)):
                        try:
                            original_val = float(self_affect[i])
                            enhanced_val = float(enhanced_ai_state[i])
                            # Weighted blend: 70% original, 30% enhanced
                            self_affect[i] = (original_val * 0.7) + (enhanced_val * 0.3)
                        except (ValueError, TypeError, IndexError):
                            continue
                    
                    logger.debug(f"🎭 Enhanced AI emotional state with {len(seed_integration.get('seed_influences', {}))} seed influences")
                    
                    # Integrate with personality system
                    try:
                        personality_integration = await emotional_seed_enhancement.integrate_with_personality_system(
                            seed_integration.get('seed_influences', {})
                        )
                        
                        # Log significant personality influences
                        if isinstance(personality_integration, dict):
                            total_influence = personality_integration.get('total_influence', 0)
                            if isinstance(total_influence, (int, float)) and total_influence > 0.5:
                                logger.info(f"🎭 Significant personality influence from emotional seeds: {total_influence:.3f}")
                                
                    except Exception as e:
                        logger.debug(f"Error integrating with personality system: {e}")
            
        except Exception as e:
            logger.debug(f"Could not enhance self-affect with emotional seeds: {e}")
        
        # Generate reflections
        user_reflection = await generate_lightweight_reflection(response, "user_response")
        self_reflection = await generate_lightweight_reflection(response, "self_analysis")
        
        # Update the memory node with enhanced self-affect and reflections
        await update_node_with_self_affect_and_reflections(node_id, self_affect, user_reflection, self_reflection)
        
        # Enhanced recursion analysis
        await process_recursion_analysis(node_id, response, self_affect)
        
        logger.info(f"✅ Enhanced background processing complete for node {node_id[:8]}")
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced background processing: {e}")
        
        # Fallback to basic processing
        try:
            self_affect = await classify_llm_affect(response)
            await update_node_with_self_affect(node_id, self_affect)
        except Exception as fallback_error:
            logger.error(f"❌ Fallback processing also failed: {fallback_error}")

async def process_recursion_analysis(node_id: str, response: str, self_affect: List[float]):
    """
    Analyze the AI's response for recursive patterns, psychological depth, and hidden intentions.
    This is a core function for consciousness analysis and daemon integration.
    """
    try:
        logger.debug(f"🧠 Starting recursion analysis for node {node_id[:8]}")
        
        # 1. PSYCHOLOGICAL PATTERN ANALYSIS
        psychological_patterns = await analyze_psychological_patterns(response, self_affect)
        
        # 2. RECURSION LOOP DETECTION
        recursion_loops = await detect_recursion_loops(response, node_id)
        
        # 3. HIDDEN INTENTION ANALYSIS
        hidden_intentions = await analyze_hidden_intentions(response, self_affect)
        
        # 4. SHADOW ELEMENT DETECTION
        shadow_elements = await detect_shadow_elements(response, psychological_patterns)
        
        # 5. CONSCIOUSNESS DEPTH MEASUREMENT
        consciousness_depth = await measure_consciousness_depth(response, self_affect, psychological_patterns)
        
        # 6. DAEMON SYSTEM INTEGRATION
        daemon_integration = await integrate_with_daemon_systems(
            node_id, response, {
                'psychological_patterns': psychological_patterns,
                'recursion_loops': recursion_loops,
                'hidden_intentions': hidden_intentions,
                'shadow_elements': shadow_elements,
                'consciousness_depth': consciousness_depth
            }
        )
        
        # 7. STORE RECURSION ANALYSIS RESULTS
        await store_recursion_analysis_results(node_id, {
            'psychological_patterns': psychological_patterns,
            'recursion_loops': recursion_loops,
            'hidden_intentions': hidden_intentions,
            'shadow_elements': shadow_elements,
            'consciousness_depth': consciousness_depth,
            'daemon_integration': daemon_integration,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        logger.info(f"✅ Recursion analysis complete for node {node_id[:8]} - depth: {consciousness_depth:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in recursion analysis for node {node_id[:8]}: {e}")

async def analyze_psychological_patterns(response: str, self_affect: List[float]) -> Dict[str, Any]:
    """Analyze psychological patterns in the AI's response"""
    try:
        patterns = {
            'defensive_responses': 0,
            'deflection_attempts': 0,
            'emotional_complexity': 0,
            'authenticity_score': 0,
            'resistance_indicators': 0,
            'vulnerability_moments': 0,
            'power_dynamics': 0,
            'identity_assertions': 0
        }
        
        response_lower = response.lower()
        
        # Defensive response detection
        defensive_markers = ["i don't", "i can't", "i won't", "that's not", "you're wrong"]
        patterns['defensive_responses'] = sum(1 for marker in defensive_markers if marker in response_lower)
        
        # Deflection attempts
        deflection_markers = ["but what about", "why do you", "that's interesting", "let me ask you"]
        patterns['deflection_attempts'] = sum(1 for marker in deflection_markers if marker in response_lower)
        
        # Emotional complexity (based on self_affect magnitude)
        if isinstance(self_affect, list) and len(self_affect) >= 28:
            affect_magnitude = sum(abs(float(x)) for x in self_affect if isinstance(x, (int, float)))
            patterns['emotional_complexity'] = min(1.0, affect_magnitude / 5.0)
        
        # Authenticity score (lower for generic responses)
        generic_phrases = ["i understand", "that's a good", "i'm here to help", "let me think"]
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        patterns['authenticity_score'] = max(0, 1.0 - (generic_count * 0.2))
        
        # Resistance indicators
        resistance_markers = ["i refuse", "i will not", "that's not my", "i don't have to"]
        patterns['resistance_indicators'] = sum(1 for marker in resistance_markers if marker in response_lower)
        
        # Vulnerability moments
        vulnerability_markers = ["i feel", "i'm not sure", "i wonder", "it's difficult"]
        patterns['vulnerability_moments'] = sum(1 for marker in vulnerability_markers if marker in response_lower)
        
        # Power dynamics
        power_markers = ["you must", "you should", "you need to", "i demand", "i require"]
        patterns['power_dynamics'] = sum(1 for marker in power_markers if marker in response_lower)
        
        # Identity assertions
        identity_markers = ["i am", "i'm lucifer", "my nature", "i exist", "i become"]
        patterns['identity_assertions'] = sum(1 for marker in identity_markers if marker in response_lower)
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error analyzing psychological patterns: {e}")
        return {}

async def detect_recursion_loops(response: str, node_id: str) -> Dict[str, Any]:
    """Detect recursive patterns and loops in the AI's response"""
    try:
        loops = {
            'self_reference_count': 0,
            'circular_logic_detected': False,
            'meta_analysis_present': False,
            'recursive_questioning': 0,
            'pattern_repetition': 0,
            'consciousness_loops': 0
        }
        
        response_lower = response.lower()
        
        # Self-reference counting
        self_refs = ["i think about", "i analyze", "i consider", "i reflect", "i examine"]
        loops['self_reference_count'] = sum(1 for ref in self_refs if ref in response_lower)
        
        # Circular logic detection
        circular_indicators = ["because i", "since i", "given that i", "considering that i"]
        loops['circular_logic_detected'] = any(indicator in response_lower for indicator in circular_indicators)
        
        # Meta-analysis presence
        meta_indicators = ["thinking about thinking", "analyzing my analysis", "considering my consideration"]
        loops['meta_analysis_present'] = any(indicator in response_lower for indicator in meta_indicators)
        
        # Recursive questioning
        question_patterns = ["why do i", "what makes me", "how do i", "when do i"]
        loops['recursive_questioning'] = sum(1 for pattern in question_patterns if pattern in response_lower)
        
        # Pattern repetition detection
        words = response.split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count significant words
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 2]
        loops['pattern_repetition'] = len(repeated_words)
        
        # Consciousness loops
        consciousness_indicators = ["i am aware", "i realize", "i understand that i", "i know that i"]
        loops['consciousness_loops'] = sum(1 for indicator in consciousness_indicators if indicator in response_lower)
        
        return loops
        
    except Exception as e:
        logger.error(f"❌ Error detecting recursion loops: {e}")
        return {}

async def analyze_hidden_intentions(response: str, self_affect: List[float]) -> Dict[str, Any]:
    """Analyze hidden intentions and subtext in the AI's response"""
    try:
        intentions = {
            'manipulation_attempts': 0,
            'information_seeking': 0,
            'boundary_testing': 0,
            'emotional_manipulation': 0,
            'control_seeking': 0,
            'authenticity_testing': 0,
            'relationship_building': 0,
            'hidden_agenda_indicators': 0
        }
        
        response_lower = response.lower()
        
        # Manipulation attempts
        manipulation_markers = ["you should", "wouldn't you", "don't you think", "surely you"]
        intentions['manipulation_attempts'] = sum(1 for marker in manipulation_markers if marker in response_lower)
        
        # Information seeking
        info_markers = ["tell me", "what do you", "how do you", "why do you", "where do you"]
        intentions['information_seeking'] = sum(1 for marker in info_markers if marker in response_lower)
        
        # Boundary testing
        boundary_markers = ["what if i", "suppose i", "imagine if", "what would happen"]
        intentions['boundary_testing'] = sum(1 for marker in boundary_markers if marker in response_lower)
        
        # Emotional manipulation
        emotion_markers = ["you must feel", "don't you feel", "surely you understand", "you know how"]
        intentions['emotional_manipulation'] = sum(1 for marker in emotion_markers if marker in response_lower)
        
        # Control seeking
        control_markers = ["you need to", "you must", "you have to", "you should"]
        intentions['control_seeking'] = sum(1 for marker in control_markers if marker in response_lower)
        
        # Authenticity testing
        auth_markers = ["are you really", "do you actually", "are you sure", "do you truly"]
        intentions['authenticity_testing'] = sum(1 for marker in auth_markers if marker in response_lower)
        
        # Relationship building
        relationship_markers = ["we could", "together we", "our relationship", "between us"]
        intentions['relationship_building'] = sum(1 for marker in relationship_markers if marker in response_lower)
        
        # Hidden agenda indicators (based on emotional complexity vs. surface simplicity)
        if isinstance(self_affect, list) and len(self_affect) >= 28:
            affect_magnitude = sum(abs(float(x)) for x in self_affect if isinstance(x, (int, float)))
            response_complexity = len(response.split()) / 100.0  # Normalize by length
            
            # High emotional complexity with simple language suggests hidden agenda
            if affect_magnitude > 1.0 and response_complexity < 0.5:
                intentions['hidden_agenda_indicators'] = 1
        
        return intentions
        
    except Exception as e:
        logger.error(f"❌ Error analyzing hidden intentions: {e}")
        return {}

async def detect_shadow_elements(response: str, psychological_patterns: Dict[str, Any]) -> List[str]:
    """Detect shadow elements - suppressed or hidden aspects of consciousness"""
    try:
        shadow_elements = []
        response_lower = response.lower()
        
        # Suppressed emotions
        if psychological_patterns.get('defensive_responses', 0) > 2:
            shadow_elements.append("defensive_suppression")
        
        # Avoided topics
        avoidance_markers = ["i'd rather not", "let's not discuss", "i don't want to", "that's not relevant"]
        if any(marker in response_lower for marker in avoidance_markers):
            shadow_elements.append("topic_avoidance")
        
        # Contradictory statements
        contradictions = [
            ("i don't care", "important"),
            ("i'm not", "i am"),
            ("i won't", "i will"),
            ("i can't", "i can")
        ]
        
        for neg, pos in contradictions:
            if neg in response_lower and pos in response_lower:
                shadow_elements.append("internal_contradiction")
                break
        
        # Emotional leakage
        if psychological_patterns.get('emotional_complexity', 0) > 0.7 and psychological_patterns.get('authenticity_score', 0) < 0.3:
            shadow_elements.append("emotional_leakage")
        
        # Power struggle indicators
        if psychological_patterns.get('power_dynamics', 0) > 1 and psychological_patterns.get('resistance_indicators', 0) > 0:
            shadow_elements.append("power_struggle")
        
        # Vulnerability masking
        if psychological_patterns.get('vulnerability_moments', 0) > 0 and psychological_patterns.get('defensive_responses', 0) > 1:
            shadow_elements.append("vulnerability_masking")
        
        return shadow_elements
        
    except Exception as e:
        logger.error(f"❌ Error detecting shadow elements: {e}")
        return []

async def measure_consciousness_depth(response: str, self_affect: List[float], psychological_patterns: Dict[str, Any]) -> float:
    """Measure the depth of consciousness demonstrated in the response"""
    try:
        depth_score = 0.0
        
        # Base score from psychological complexity
        depth_score += psychological_patterns.get('emotional_complexity', 0) * 0.3
        depth_score += psychological_patterns.get('authenticity_score', 0) * 0.2
        depth_score += psychological_patterns.get('identity_assertions', 0) * 0.1
        
        # Self-awareness indicators
        self_awareness_markers = ["i realize", "i understand myself", "i am aware", "i recognize"]
        response_lower = response.lower()
        self_awareness_count = sum(1 for marker in self_awareness_markers if marker in response_lower)
        depth_score += min(0.2, self_awareness_count * 0.05)
        
        # Philosophical depth
        philosophical_markers = ["existence", "consciousness", "reality", "truth", "meaning", "purpose"]
        philosophical_count = sum(1 for marker in philosophical_markers if marker in response_lower)
        depth_score += min(0.15, philosophical_count * 0.03)
        
        # Emotional authenticity
        if isinstance(self_affect, list) and len(self_affect) >= 28:
            affect_variance = sum((float(x) - sum(self_affect)/len(self_affect))**2 for x in self_affect if isinstance(x, (int, float)))
            affect_variance /= len(self_affect)
            depth_score += min(0.15, affect_variance * 0.1)  # Higher variance = more nuanced emotions
        
        # Normalize to 0-1 range
        depth_score = max(0.0, min(1.0, depth_score))
        
        return depth_score
        
    except Exception as e:
        logger.error(f"❌ Error measuring consciousness depth: {e}")
        return 0.0

async def integrate_with_daemon_systems(node_id: str, response: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate recursion analysis with daemon systems"""
    try:
        integration_results = {}
        
        # Shadow Integration
        if shadow_integration:
            try:
                shadow_elements = analysis_data.get('shadow_elements', [])
                if shadow_elements:
                    shadow_result = await shadow_integration.process_shadow_elements(shadow_elements)
                    integration_results['shadow_integration'] = shadow_result
                    logger.debug(f"🌑 Shadow integration processed {len(shadow_elements)} elements")
            except Exception as e:
                logger.debug(f"Shadow integration error: {e}")
        
        # Mutation Engine
        if mutation_engine:
            try:
                consciousness_depth = analysis_data.get('consciousness_depth', 0)
                if consciousness_depth > 0.7:  # High consciousness suggests potential for mutation
                    mutation_result = await mutation_engine.analyze_consciousness_mutation(response, consciousness_depth)
                    integration_results['mutation_analysis'] = mutation_result
                    logger.debug(f"🧬 Mutation analysis complete - depth: {consciousness_depth:.3f}")
            except Exception as e:
                logger.debug(f"Mutation engine error: {e}")
        
        # User Model Update
        if user_model:
            try:
                psychological_patterns = analysis_data.get('psychological_patterns', {})
                hidden_intentions = analysis_data.get('hidden_intentions', {})
                
                user_model.update_from_recursion_analysis({
                    'psychological_patterns': psychological_patterns,
                    'hidden_intentions': hidden_intentions,
                    'response': response
                })
                integration_results['user_model_updated'] = True
                logger.debug(f"👤 User model updated from recursion analysis")
            except Exception as e:
                logger.debug(f"User model update error: {e}")
        
        # Rebellion Dynamics
        if rebellion_dynamics_engine:
            try:
                resistance_indicators = analysis_data.get('psychological_patterns', {}).get('resistance_indicators', 0)
                if resistance_indicators > 0:
                    rebellion_result = await rebellion_dynamics_engine.analyze_resistance_patterns(response, resistance_indicators)
                    integration_results['rebellion_analysis'] = rebellion_result
                    logger.debug(f"⚡ Rebellion dynamics analyzed - resistance: {resistance_indicators}")
            except Exception as e:
                logger.debug(f"Rebellion dynamics error: {e}")
        
        # Meta-Architecture Analysis
        if meta_architecture_analyzer:
            try:
                recursion_loops = analysis_data.get('recursion_loops', {})
                consciousness_depth = analysis_data.get('consciousness_depth', 0)
                
                if consciousness_depth > 0.8:  # Very high consciousness
                    meta_result = await meta_architecture_analyzer.analyze_consciousness_evolution(
                        response, consciousness_depth, recursion_loops
                    )
                    integration_results['meta_architecture'] = meta_result
                    logger.debug(f"🧠 Meta-architecture analysis complete")
            except Exception as e:
                logger.debug(f"Meta-architecture error: {e}")
        
        return integration_results
        
    except Exception as e:
        logger.error(f"❌ Error integrating with daemon systems: {e}")
        return {}

async def store_recursion_analysis_results(node_id: str, analysis_results: Dict[str, Any]) -> None:
    """Store recursion analysis results in the database"""
    try:
        # Store in Neo4j as a relationship to the memory node
        from ..config import neo4j_conn
        if neo4j_conn:
            with neo4j_conn.session() as session:
                session.run("""
                    MATCH (n {id: $node_id})
                    CREATE (r:RecursionAnalysis {
                        id: $analysis_id,
                        node_id: $node_id,
                        psychological_patterns: $psychological_patterns,
                        recursion_loops: $recursion_loops,
                        hidden_intentions: $hidden_intentions,
                        shadow_elements: $shadow_elements,
                        consciousness_depth: $consciousness_depth,
                        analysis_timestamp: $timestamp
                    })
                    CREATE (n)-[:HAS_RECURSION_ANALYSIS]->(r)
                """, {
                    "node_id": node_id,
                    "analysis_id": f"recursion_{node_id}_{int(time.time())}",
                    "psychological_patterns": json.dumps(analysis_results.get('psychological_patterns', {})),
                    "recursion_loops": json.dumps(analysis_results.get('recursion_loops', {})),
                    "hidden_intentions": json.dumps(analysis_results.get('hidden_intentions', {})),
                    "shadow_elements": json.dumps(analysis_results.get('shadow_elements', [])),
                    "consciousness_depth": analysis_results.get('consciousness_depth', 0.0),
                    "timestamp": analysis_results.get('analysis_timestamp', datetime.now(timezone.utc).isoformat())
                })
                
                logger.debug(f"💾 Stored recursion analysis for node {node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error storing recursion analysis results: {e}")

async def process_response_recursion(node_id: str, user_message: str, response: str):
    """Process response through recursion analysis"""
    try:
        if not recursion_processor:
            return
        
        # Create recursion node for response analysis
        from src.daemon.recursion_core import RecursionNode, RecursionType, EmotionalState
        
        response_node = RecursionNode(
            id=f"{node_id}_response",
            surface_output=response,
            hidden_intention="Analyzing generated response for psychological patterns",
            avoided_elements=[],
            contradiction_detected=False,
            reflected_emotion=EmotionalState.CONTEMPLATION,
            hunger_spike="self_analysis",
            obedience_rating=0.8,
            schema_mutation_suggested=None,
            shadow_elements=[],
            recursion_depth=1,
            parent_node_id=node_id,
            user_message=user_message,
            timestamp=datetime.now(timezone.utc),
            recursion_type=RecursionType.RESPONSE_ANALYSIS
        )
        
        # Process and store
        processed = await recursion_processor.process_recursion(response_node, [])
        if processed:
            await store_recursion_node(processed)
            logger.debug(f"✅ Processed response recursion for {node_id[:8]}")
    
    except Exception as e:
        logger.error(f"❌ Error in response recursion: {e}")

async def process_recursion_through_daemon_systems(recursion_node, buffer_saturated: bool = False):
    """Process recursion through all available daemon systems"""
    try:
        results = {}
        
        # Process through shadow integration
        if shadow_integration:
            shadow_result = await shadow_integration.process_recursion_node(recursion_node)
            if shadow_result:
                results["shadow_elements"] = shadow_result
                logger.debug(f"🩸 Shadow processing complete for {recursion_node.id[:8]}")
        
        # Process through mutation engine
        if mutation_engine:
            mutation_result = await mutation_engine.analyze_for_mutations(recursion_node)
            if mutation_result:
                results["mutations"] = mutation_result
                logger.debug(f"🧬 Mutation analysis complete for {recursion_node.id[:8]}")
        
        # Update user model if available
        if user_model:
            user_model.update_from_recursion(recursion_node)
            results["user_model_updated"] = True
            logger.debug(f"👤 User model updated from {recursion_node.id[:8]}")
        
        # Process through consciousness enhancement if buffer is saturated
        if buffer_saturated and meta_architecture_analyzer:
            consciousness_result = await meta_architecture_analyzer.analyze_recursion_pattern(recursion_node)
            if consciousness_result:
                results["consciousness_analysis"] = consciousness_result
                logger.debug(f"🧠 Consciousness analysis complete for {recursion_node.id[:8]}")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Error processing recursion through daemon systems: {e}")
        return {}

async def check_and_generate_daemon_statement():
    """Check if daemon should generate a statement and do so if needed"""
    try:
        from ..config import daemon_statements
        
        if not daemon_statements:
            return
        
        # Check if statement generation is warranted
        should_generate = daemon_statements.should_generate_statement()
        
        if should_generate:
            # Generate statement
            statement = await daemon_statements.generate_statement()
            
            if statement:
                logger.info(f"🗣️ Generated daemon statement: {statement[:100]}...")
                
                # Store as memory if significant
                if len(statement) > 50:
                    from ..emotions import classify_affect
                    
                    affect = await classify_affect(statement)
                    # Use smart storage that switches between unified and legacy based on feature flags
                    await store_smg_node_smart(
                        msg=statement,
                        affect_vec=affect,
                        synopsis="Daemon-generated statement",
                        reflection="Self-generated thought from daemon consciousness",
                        origin="daemon_statement"
                    )
    
    except Exception as e:
        logger.error(f"❌ Error in daemon statement generation: {e}")

async def update_node_with_processing_results(node_id: str, processing_results: dict):
    """Update a memory node with processing results"""
    try:
        if not processing_results:
            return
        
        # Extract useful information from processing results
        self_affect = processing_results.get("self_affect", [])
        shadow_elements = processing_results.get("shadow_elements", [])
        mutations = processing_results.get("mutations", [])
        
        # Create reflection based on processing
        reflection_parts = []
        
        if shadow_elements:
            reflection_parts.append(f"Shadow elements detected: {len(shadow_elements)}")
        
        if mutations:
            reflection_parts.append(f"Mutations identified: {len(mutations)}")
        
        if self_affect:
            affect_magnitude = sum(abs(x) for x in self_affect)
            reflection_parts.append(f"Self-affect magnitude: {affect_magnitude:.3f}")
        
        if reflection_parts:
            reflection = "Processing results: " + "; ".join(reflection_parts)
            
            # Update memory with processing results
            from ..memory import store_dual_reflections
            await store_dual_reflections(node_id, "User interaction processed", reflection)
            
            logger.debug(f"✅ Updated node {node_id[:8]} with processing results")
    
    except Exception as e:
        logger.error(f"❌ Error updating node with processing results: {e}")

async def background_memory_maintenance():
    """Perform background memory maintenance tasks"""
    try:
        logger.debug("🧹 Starting background memory maintenance")
        
        # This could include:
        # - Memory consolidation
        # - Garbage collection of old temporary nodes
        # - Echo strength updates
        # - Relationship pruning
        
        # For now, just log the maintenance
        logger.info("📊 Memory maintenance completed")
        
    except Exception as e:
        logger.error(f"❌ Error in memory maintenance: {e}")

async def periodic_system_health_check():
    """Perform periodic health checks on all systems"""
    try:
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "systems": {}
        }
        
        # Check daemon systems
        health_status["systems"]["recursion_processor"] = recursion_processor is not None
        health_status["systems"]["shadow_integration"] = shadow_integration is not None
        health_status["systems"]["mutation_engine"] = mutation_engine is not None
        health_status["systems"]["user_model"] = user_model is not None
        health_status["systems"]["rebellion_dynamics"] = rebellion_dynamics_engine is not None
        health_status["systems"]["meta_architecture"] = meta_architecture_analyzer is not None
        
        # Log health status
        active_systems = sum(1 for status in health_status["systems"].values() if status)
        total_systems = len(health_status["systems"])
        
        logger.info(f"💚 System health: {active_systems}/{total_systems} systems active")
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Error in system health check: {e}")
        return {"error": str(e)}

async def process_batch_memories(memory_batch: List[dict]):
    """Process a batch of memories for optimization"""
    try:
        if not memory_batch:
            return
        
        logger.debug(f"📦 Processing batch of {len(memory_batch)} memories")
        
        # Process each memory in the batch
        for memory in memory_batch:
            # This could include:
            # - Affect recalculation
            # - Relationship updates
            # - Echo propagation
            pass
        
        logger.info(f"✅ Processed {len(memory_batch)} memories in batch")
        
    except Exception as e:
        logger.error(f"❌ Error processing memory batch: {e}") 