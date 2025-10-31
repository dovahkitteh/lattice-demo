"""
Self-awareness system for the Lattice AI

This module provides comprehensive self-awareness capabilities including:
- Auto-updating feature detection
- Self-reflection endpoint management
- Feature changelog tracking
- CLAUDE.md auto-documentation
"""

from .auto_update import (
    auto_updater,
    register_new_feature,
    update_feature_status,
    get_feature_changelog,
    update_claude_md
)

__all__ = [
    "auto_updater",
    "register_new_feature", 
    "update_feature_status",
    "get_feature_changelog",
    "update_claude_md"
]