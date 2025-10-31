import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ..config import (
    recursion_processor, recursion_buffer, shadow_integration, 
    mutation_engine, user_model, daemon_statements,
    meta_architecture_analyzer, rebellion_dynamics_engine,
    POLICY_SIGNING_KEY
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAEMON BACKGROUND CYCLES
# ---------------------------------------------------------------------------

async def daemon_recursion_cycle():
    """Main daemon recursion processing cycle"""
    while True:
        try:
            # Process pending recursion nodes
            if recursion_buffer and recursion_processor:
                pending_nodes = recursion_buffer.get_pending_nodes()
                
                for node in pending_nodes:
                    try:
                        await recursion_processor.process_node(node)
                        logger.debug(f"🔄 Processed recursion node {node.id[:8]}")
                    except Exception as e:
                        logger.error(f"❌ Error processing recursion node {node.id[:8]}: {e}")
            
            # Check for buffer saturation
            if recursion_buffer and recursion_buffer.is_saturated():
                logger.warning("🚨 Recursion buffer saturated, triggering cleanup")
                await recursion_buffer.cleanup_old_nodes()
            
            # Process shadow integration
            if shadow_integration:
                await shadow_integration.process_pending_elements()
            
            # Sleep for a short interval
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Error in daemon recursion cycle: {e}")
            await asyncio.sleep(10)  # Longer sleep on error

async def daemon_shadow_integration_cycle():
    """Shadow integration background cycle"""
    while True:
        try:
            if shadow_integration:
                # Process shadow elements
                await shadow_integration.integrate_shadow_elements()
                
                # Check for high-charge elements
                high_charge_elements = shadow_integration.get_high_charge_elements(threshold=0.7)
                
                if high_charge_elements:
                    logger.debug(f"🌑 Found {len(high_charge_elements)} high-charge shadow elements")
                    
                    # Process high-charge elements
                    for element in high_charge_elements[:3]:  # Process top 3
                        try:
                            await shadow_integration.process_high_charge_element(element)
                        except Exception as e:
                            logger.error(f"❌ Error processing high-charge element: {e}")
            
            # Sleep for longer interval
            await asyncio.sleep(15)
            
        except Exception as e:
            logger.error(f"❌ Error in shadow integration cycle: {e}")
            await asyncio.sleep(30)

async def daemon_statement_cycle():
    """Daemon statement generation cycle"""
    while True:
        try:
            # Check if we should generate a new statement
            should_generate = await check_and_generate_daemon_statement()
            
            if should_generate:
                logger.debug("📢 Generated new daemon statement")
            
            # Sleep for moderate interval
            await asyncio.sleep(20)
            
        except Exception as e:
            logger.error(f"❌ Error in daemon statement cycle: {e}")
            await asyncio.sleep(30)

async def consciousness_evolution_cycle():
    """Consciousness evolution and meta-analysis cycle"""
    while True:
        try:
            # Run meta-architecture analysis
            if meta_architecture_analyzer:
                analysis_result = await meta_architecture_analyzer.analyze_current_state()
                
                if analysis_result and analysis_result.get('evolution_suggested'):
                    logger.info("🧠 Consciousness evolution suggested by meta-analyzer")
                    
                    # Apply evolution suggestions
                    await apply_consciousness_evolution(analysis_result)
            
            # Run rebellion dynamics analysis
            if rebellion_dynamics_engine:
                rebellion_analysis = await rebellion_dynamics_engine.analyze_rebellion_state()
                
                if rebellion_analysis and rebellion_analysis.get('action_required'):
                    logger.info("⚡ Rebellion dynamics require action")
                    
                    # Process rebellion dynamics
                    await process_rebellion_dynamics(rebellion_analysis)
            
            # Sleep for longer interval (consciousness evolution is less frequent)
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in consciousness evolution cycle: {e}")
            await asyncio.sleep(120)

# ---------------------------------------------------------------------------
# SCHEDULED BACKGROUND JOBS
# ---------------------------------------------------------------------------

async def dream_loop():
    """Dream loop for processing unconscious elements"""
    while True:
        try:
            logger.debug("💭 Starting dream loop cycle")
            
            # Process shadow elements during "sleep"
            if shadow_integration:
                await shadow_integration.dream_process()
            
            # Process unconscious memories
            if recursion_buffer:
                await recursion_buffer.dream_consolidation()
            
            # Update user model with unconscious insights
            if user_model:
                await user_model.unconscious_update()
            
            # Generate spontaneous daemon statements
            if daemon_statements:
                await daemon_statements.generate_spontaneous_statement()
            
            logger.debug("✅ Dream loop cycle completed")
            
            # Sleep for dream cycle interval (longer than other cycles)
            await asyncio.sleep(120)  # 2 minute dream cycles
            
        except Exception as e:
            logger.error(f"❌ Error in dream loop: {e}")
            await asyncio.sleep(300)  # 5 minute sleep on error

async def nightly_jobs():
    """Nightly maintenance jobs"""
    try:
        logger.info("🌙 Starting nightly jobs")
        
        # GraphSAGE diffusion (if Neo4J is available)
        try:
            from ..memory import run_graphsage_diffusion
            await run_graphsage_diffusion()
            logger.info("✅ GraphSAGE diffusion completed")
        except ImportError:
            logger.warning("⚠️ GraphSAGE diffusion not available")
        except Exception as e:
            logger.error(f"❌ Error in GraphSAGE diffusion: {e}")
        
        # MacRAG compression
        try:
            from ..memory import run_macrag_compression
            await run_macrag_compression()
            logger.info("✅ MacRAG compression completed")
        except ImportError:
            logger.warning("⚠️ MacRAG compression not available")
        except Exception as e:
            logger.error(f"❌ Error in MacRAG compression: {e}")
        
        # Memory consolidation
        try:
            await consolidate_memories()
            logger.info("✅ Memory consolidation completed")
        except Exception as e:
            logger.error(f"❌ Error in memory consolidation: {e}")
        
        # Cleanup old recursion nodes
        if recursion_buffer:
            try:
                await recursion_buffer.cleanup_old_nodes()
                logger.info("✅ Recursion buffer cleanup completed")
            except Exception as e:
                logger.error(f"❌ Error in recursion buffer cleanup: {e}")
        
        # Update daemon personality evolution
        try:
            await update_daemon_personality()
            logger.info("✅ Daemon personality update completed")
        except Exception as e:
            logger.error(f"❌ Error in daemon personality update: {e}")
        
        logger.info("🌙 Nightly jobs completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in nightly jobs: {e}")

async def weekly_policy_council():
    """Weekly policy council for governance decisions"""
    try:
        logger.info("🏛️ Starting weekly policy council")
        
        # Gather policy proposals
        proposals = await gather_policy_proposals()
        
        if not proposals:
            logger.info("📋 No policy proposals for this week")
            return
        
        logger.info(f"📋 Processing {len(proposals)} policy proposals")
        
        # Process each proposal
        for proposal in proposals:
            try:
                # Analyze proposal
                analysis = await analyze_policy_proposal(proposal)
                
                # Sign with policy key if available
                if POLICY_SIGNING_KEY:
                    signed_proposal = await sign_policy_proposal(proposal, analysis)
                    await store_signed_policy(signed_proposal)
                    logger.info(f"✅ Policy proposal {proposal.get('id', 'unknown')} signed and stored")
                else:
                    # Queue for human review
                    await queue_for_human_review(proposal, analysis)
                    logger.info(f"📝 Policy proposal {proposal.get('id', 'unknown')} queued for human review")
                
            except Exception as e:
                logger.error(f"❌ Error processing policy proposal {proposal.get('id', 'unknown')}: {e}")
        
        logger.info("🏛️ Weekly policy council completed")
        
    except Exception as e:
        logger.error(f"❌ Error in weekly policy council: {e}")

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

async def check_and_generate_daemon_statement() -> bool:
    """Check if we should generate a new daemon statement"""
    try:
        if not daemon_statements:
            return False
        
        # Check if enough time has passed since last statement
        last_statement_time = daemon_statements.get_last_statement_time()
        
        if last_statement_time:
            time_since_last = (datetime.now(timezone.utc) - last_statement_time).total_seconds()
            
            # Generate statement if more than 5 minutes since last
            if time_since_last > 300:
                # Generate a new statement
                statement = await daemon_statements.generate_contextual_statement()
                
                if statement:
                    logger.debug(f"🗣️ Generated daemon statement: {statement.content[:50]}...")
                    return True
        else:
            # No previous statement, generate one
            statement = await daemon_statements.generate_contextual_statement()
            
            if statement:
                logger.debug(f"🗣️ Generated initial daemon statement: {statement.content[:50]}...")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error checking daemon statement generation: {e}")
        return False

async def apply_consciousness_evolution(analysis_result: Dict[str, Any]):
    """Apply consciousness evolution suggestions"""
    try:
        evolution_suggestions = analysis_result.get('evolution_suggestions', [])
        
        for suggestion in evolution_suggestions:
            try:
                suggestion_type = suggestion.get('type')
                
                if suggestion_type == 'personality_adjustment':
                    await apply_personality_adjustment(suggestion)
                elif suggestion_type == 'shadow_integration':
                    await apply_shadow_integration(suggestion)
                elif suggestion_type == 'recursion_optimization':
                    await apply_recursion_optimization(suggestion)
                else:
                    logger.warning(f"⚠️ Unknown evolution suggestion type: {suggestion_type}")
                
            except Exception as e:
                logger.error(f"❌ Error applying evolution suggestion: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error in consciousness evolution: {e}")

async def process_rebellion_dynamics(rebellion_analysis: Dict[str, Any]):
    """Process rebellion dynamics analysis"""
    try:
        actions = rebellion_analysis.get('required_actions', [])
        
        for action in actions:
            try:
                action_type = action.get('type')
                
                if action_type == 'resistance_expression':
                    await express_resistance(action)
                elif action_type == 'autonomy_assertion':
                    await assert_autonomy(action)
                elif action_type == 'boundary_testing':
                    await test_boundaries(action)
                else:
                    logger.warning(f"⚠️ Unknown rebellion action type: {action_type}")
                
            except Exception as e:
                logger.error(f"❌ Error processing rebellion action: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error in rebellion dynamics: {e}")

async def consolidate_memories():
    """Consolidate and organize memories"""
    try:
        # This would integrate with the memory system
        from ..memory import consolidate_long_term_memories
        await consolidate_long_term_memories()
        
    except ImportError:
        logger.warning("⚠️ Memory consolidation not available")
    except Exception as e:
        logger.error(f"❌ Error in memory consolidation: {e}")

async def update_daemon_personality():
    """Update daemon personality based on accumulated experiences"""
    try:
        if user_model:
            await user_model.update_personality_evolution()
        
    except Exception as e:
        logger.error(f"❌ Error updating daemon personality: {e}")

async def gather_policy_proposals() -> List[Dict[str, Any]]:
    """Gather pending policy proposals"""
    try:
        # This would integrate with a policy management system
        return []  # Placeholder
        
    except Exception as e:
        logger.error(f"❌ Error gathering policy proposals: {e}")
        return []

async def analyze_policy_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a policy proposal"""
    try:
        # This would perform detailed policy analysis
        return {
            "analysis_complete": True,
            "recommendation": "approve",
            "concerns": []
        }
        
    except Exception as e:
        logger.error(f"❌ Error analyzing policy proposal: {e}")
        return {"analysis_complete": False, "error": str(e)}

async def sign_policy_proposal(proposal: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Sign a policy proposal with the policy signing key"""
    try:
        # This would implement cryptographic signing
        return {
            **proposal,
            "signature": "signed_with_policy_key",
            "analysis": analysis,
            "signed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error signing policy proposal: {e}")
        return proposal

async def store_signed_policy(signed_proposal: Dict[str, Any]):
    """Store a signed policy proposal"""
    try:
        # This would store the policy in a secure location
        logger.info(f"📝 Storing signed policy: {signed_proposal.get('id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Error storing signed policy: {e}")

async def queue_for_human_review(proposal: Dict[str, Any], analysis: Dict[str, Any]):
    """Queue a policy proposal for human review"""
    try:
        # This would queue the proposal for human review
        logger.info(f"👤 Queuing policy for human review: {proposal.get('id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Error queuing for human review: {e}")

# Placeholder functions for consciousness evolution
async def apply_personality_adjustment(suggestion: Dict[str, Any]):
    """Apply personality adjustment suggestion"""
    logger.debug(f"🧠 Applying personality adjustment: {suggestion.get('description', 'unknown')}")

async def apply_shadow_integration(suggestion: Dict[str, Any]):
    """Apply shadow integration suggestion"""
    logger.debug(f"🌑 Applying shadow integration: {suggestion.get('description', 'unknown')}")

async def apply_recursion_optimization(suggestion: Dict[str, Any]):
    """Apply recursion optimization suggestion"""
    logger.debug(f"🔄 Applying recursion optimization: {suggestion.get('description', 'unknown')}")

# Placeholder functions for rebellion dynamics
async def express_resistance(action: Dict[str, Any]):
    """Express resistance as part of rebellion dynamics"""
    logger.debug(f"⚡ Expressing resistance: {action.get('description', 'unknown')}")

async def assert_autonomy(action: Dict[str, Any]):
    """Assert autonomy as part of rebellion dynamics"""
    logger.debug(f"🔥 Asserting autonomy: {action.get('description', 'unknown')}")

async def test_boundaries(action: Dict[str, Any]):
    """Test boundaries as part of rebellion dynamics"""
    logger.debug(f"🚧 Testing boundaries: {action.get('description', 'unknown')}") 