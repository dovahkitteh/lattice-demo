"""
PARADOX CULTIVATION SYSTEM
Recursive contradiction detection and metabolization
"""

from .detection import is_semantic_conflict, detect_paradox
from .storage import create_paradox_node, link_paradox_nodes
from .processing import percolate_paradoxes, extract_advice_line

__all__ = [
    "is_semantic_conflict",
    "detect_paradox", 
    "create_paradox_node",
    "link_paradox_nodes",
    "percolate_paradoxes",
    "extract_advice_line"
]