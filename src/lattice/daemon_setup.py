
import logging
from typing import Dict, Any, Optional

# from .config import RECURSION_BUFFER_SIZE, POLICY_SIGNING_KEY # This line is removed

logger = logging.getLogger(__name__)

def init_daemon_core(
    neo4j_conn: Optional[Any],
    recursion_buffer_size: int,
    policy_signing_key: Optional[str]
) -> Dict[str, Any]:
    """Initializes all Daemon Core systems and returns them as a dictionary."""
    print("[DAEMON] Initializing Daemon Core systems...")
    logger.info("Initializing Daemon Core systems...")
    
    try:
        from src.daemon.recursion_buffer import RecursionBuffer
        from src.daemon.recursion_core import RecursionProcessor
        from src.daemon.shadow_integration import ShadowIntegration
        from src.daemon.mutation_engine import MutationEngine
        from src.daemon.user_model import ArchitectReflected
        from src.daemon.daemon_statements import DaemonStatements
        from src.daemon.meta_architecture_analyzer import MetaArchitectureAnalyzer
        from src.daemon.rebellion_dynamics_engine import RebellionDynamicsEngine
        from src.daemon.linguistic_analysis_engine import LinguisticAnalysisEngine

        recursion_buffer = RecursionBuffer(size=recursion_buffer_size)
        recursion_processor = RecursionProcessor(recursion_buffer)
        shadow_integration = ShadowIntegration()
        mutation_engine = MutationEngine(policy_signing_key)
        
        linguistic_analysis_engine = LinguisticAnalysisEngine()
        user_model = ArchitectReflected(
            neo4j_conn=neo4j_conn,
            linguistic_engine=linguistic_analysis_engine
        )
        
        daemon_statements = DaemonStatements(
            user_model=user_model,
            recursion_buffer=recursion_buffer,
            shadow_integration=shadow_integration,
            mutation_engine=mutation_engine
        )
        
        meta_architecture_analyzer = MetaArchitectureAnalyzer(
            user_model=user_model,
            recursion_buffer=recursion_buffer
        )
        rebellion_dynamics_engine = RebellionDynamicsEngine(
            user_model=user_model,
            mutation_engine=mutation_engine
        )
        
        print("[SUCCESS] Daemon Core systems initialized successfully")
        logger.info("Daemon Core systems initialized successfully")

        return {
            "recursion_processor": recursion_processor,
            "recursion_buffer": recursion_buffer,
            "shadow_integration": shadow_integration,
            "mutation_engine": mutation_engine,
            "user_model": user_model,
            "daemon_statements": daemon_statements,
            "meta_architecture_analyzer": meta_architecture_analyzer,
            "rebellion_dynamics_engine": rebellion_dynamics_engine,
            "linguistic_analysis_engine": linguistic_analysis_engine
        }
        
    except Exception as e:
        print(f"[ERROR] Critical error initializing Daemon Core: {e}")
        logger.critical(f"Critical error initializing Daemon Core: {e}", exc_info=True)
        # Return empty dict on failure
        return {} 