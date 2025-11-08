# Memory System Package
# Handles all memory storage, retrieval, and echo functionality

from .storage import (
    store_smg_node,
    store_dual_affect_node,
    store_dual_affect_node_with_id,
    update_node_with_self_affect,
    update_node_with_self_affect_and_reflections,
    store_dual_reflections,
    store_recursion_node
)

from .retrieval import (
    retrieve_context,
    retrieve_context_with_compression,
    compress_context_memories,
    get_recent_memories,
    get_recent_echoes,
    get_context_for_query,
    get_memory_stats
)

from .echo import (
    echo_update,
    echo_update_unified,
    create_echo_relationship,
    get_echo_statistics,
    get_enhanced_echo_statistics,
    analyze_memory_access_patterns
)

# New unified storage system - Phase 1
from .memory_node import (
    MemoryNode,
    MemoryNodeType,
    MemoryOrigin,
    MemoryLifecycleState
)

from .unified_storage import (
    store_memory_node,
    create_and_store_smg_memory,
    create_and_store_dual_affect_memory,
    create_and_store_dual_affect_memory_with_id,
    update_memory_node_affect,
    update_memory_node_reflections,
    store_smg_node_unified,
    store_dual_affect_node_unified,
    store_dual_affect_node_with_id_unified,
    validate_unified_storage,
    get_unified_storage_diagnostics,
    unified_storage
)

# Emotional memory seed storage
from .emotional_seed_storage import (
    store_emotional_memory_seed,
    emotional_seed_storage
)

# Emotional memory seed enhancement system
from .emotional_seed_enhancement import (
    emotional_seed_enhancement
)

# Feature flags for gradual rollout
ENABLE_UNIFIED_STORAGE = True
VALIDATE_AGAINST_LEGACY = True
LEGACY_FALLBACK_ENABLED = True

def configure_unified_storage(use_unified: bool = True, validate_legacy: bool = True, fallback_enabled: bool = True):
    """Configure the unified storage system feature flags"""
    global ENABLE_UNIFIED_STORAGE, VALIDATE_AGAINST_LEGACY, LEGACY_FALLBACK_ENABLED
    ENABLE_UNIFIED_STORAGE = use_unified
    VALIDATE_AGAINST_LEGACY = validate_legacy
    LEGACY_FALLBACK_ENABLED = fallback_enabled
    
    # Update the unified storage instance
    unified_storage.use_new_storage = use_unified
    unified_storage.validate_against_legacy = validate_legacy
    unified_storage.legacy_fallback_enabled = fallback_enabled

def get_storage_config():
    """Get current storage configuration"""
    return {
        "unified_storage_enabled": ENABLE_UNIFIED_STORAGE,
        "validate_against_legacy": VALIDATE_AGAINST_LEGACY,
        "legacy_fallback_enabled": LEGACY_FALLBACK_ENABLED
    }

# Smart storage functions that use unified storage when enabled
async def store_smg_node_smart(msg: str, affect_vec: list[float], synopsis: str, 
                              reflection: str | None = None, origin: str = "external_user") -> str:
    """Smart SMG storage that uses unified storage when enabled"""
    if ENABLE_UNIFIED_STORAGE:
        return await store_smg_node_unified(msg, affect_vec, synopsis, reflection, origin)
    else:
        return await store_smg_node(msg, affect_vec, synopsis, reflection, origin)

async def store_dual_affect_node_smart(msg: str, user_affect: list[float], self_affect: list[float], 
                                      synopsis: str, reflection: str | None = None, 
                                      origin: str = "dual_channel", session_id: str | None = None, 
                                      turn_id: str | None = None) -> str:
    """Smart dual-affect storage that uses unified storage when enabled"""
    if ENABLE_UNIFIED_STORAGE:
        return await store_dual_affect_node_unified(msg, user_affect, self_affect, synopsis, reflection, origin, session_id, turn_id)
    else:
        return await store_dual_affect_node(msg, user_affect, self_affect, synopsis, reflection, origin)

async def store_dual_affect_node_with_id_smart(node_id: str, msg: str, user_affect: list[float], 
                                              self_affect: list[float], synopsis: str,
                                              reflection: str | None = None, 
                                              origin: str = "dual_channel") -> str:
    """Smart dual-affect storage with ID that uses unified storage when enabled"""
    if ENABLE_UNIFIED_STORAGE:
        return await store_dual_affect_node_with_id_unified(node_id, msg, user_affect, self_affect, synopsis, reflection, origin)
    else:
        return await store_dual_affect_node_with_id(node_id, msg, user_affect, self_affect, synopsis, reflection, origin)

__all__ = [
    # Legacy storage functions
    'store_smg_node',
    'store_dual_affect_node',
    'store_dual_affect_node_with_id',
    'update_node_with_self_affect',
    'update_node_with_self_affect_and_reflections',
    'store_dual_reflections',
    'store_recursion_node',
    
    # Smart storage functions (feature-flagged)
    'store_smg_node_smart',
    'store_dual_affect_node_smart',
    'store_dual_affect_node_with_id_smart',
    
    # Unified storage system - Phase 1
    'MemoryNode',
    'MemoryNodeType',
    'MemoryOrigin',
    'MemoryLifecycleState',
    'store_memory_node',
    'create_and_store_smg_memory',
    'create_and_store_dual_affect_memory',
    'create_and_store_dual_affect_memory_with_id',
    'update_memory_node_affect',
    'update_memory_node_reflections',
    'store_smg_node_unified',
    'store_dual_affect_node_unified',
    'store_dual_affect_node_with_id_unified',
    'validate_unified_storage',
    'get_unified_storage_diagnostics',
    'unified_storage',
    
    # Emotional memory seed storage
    'store_emotional_memory_seed',
    'emotional_seed_storage',
    
    # Emotional memory seed enhancement system
    'emotional_seed_enhancement',
    
    # Configuration functions
    'configure_unified_storage',
    'get_storage_config',
    
    # Retrieval functions
    'retrieve_context',
    'retrieve_context_with_compression',
    'compress_context_memories',
    'get_recent_memories',
    'get_recent_echoes',
    'get_context_for_query',
    'get_memory_stats',
    
    # Echo functions
    'echo_update',
    'echo_update_unified',
    'create_echo_relationship',
    'get_echo_statistics',
    'get_enhanced_echo_statistics',
    'analyze_memory_access_patterns'
]
