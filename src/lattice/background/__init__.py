# src/lattice/background/__init__.py
"""
Background processing, daemon cycles, and reflection for the Lattice service.
"""

from .processing import (
    process_conversation_turn,
    process_completed_response_recursion
)
from .daemon_cycles import (
    daemon_recursion_cycle,
    daemon_shadow_integration_cycle,
    daemon_statement_cycle,
    consciousness_evolution_cycle,
    dream_loop,
    nightly_jobs,
    weekly_policy_council
)
from .reflection import reflect_on_turn

__all__ = [
    # Processing
    "process_conversation_turn",
    "process_completed_response_recursion",

    # Daemon Cycles
    "daemon_recursion_cycle",
    "daemon_shadow_integration_cycle",
    "daemon_statement_cycle",
    "consciousness_evolution_cycle",
    "dream_loop",

    # Scheduled Jobs
    "nightly_jobs",
    "weekly_policy_council",

    # Reflection
    "reflect_on_turn"
]
