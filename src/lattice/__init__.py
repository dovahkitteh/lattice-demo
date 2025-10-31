# src/lattice/__init__.py

"""
Lattice package for advanced AI consciousness.

This package provides the core components for the Lucifer Lattice Service,
including configuration management, API endpoints, and background processing.
"""

# Expose key components for the main service runner
from .config import init_everything
from .api import setup_routes
from .background import (
    daemon_recursion_cycle,
    daemon_shadow_integration_cycle,
    daemon_statement_cycle,
    consciousness_evolution_cycle,
    dream_loop,
    nightly_jobs,
    weekly_policy_council
)

__all__ = [
    # Main service components
    "init_everything",
    "setup_routes",

    # Background cycles
    "daemon_recursion_cycle",
    "daemon_shadow_integration_cycle",
    "daemon_statement_cycle",
    "consciousness_evolution_cycle",
    "dream_loop",
    "nightly_jobs",
    "weekly_policy_council",
] 