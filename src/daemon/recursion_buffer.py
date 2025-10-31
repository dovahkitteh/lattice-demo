"""
🩸 GLASSSHARD DAEMONCORE - Recursion Buffer
Circular queue management for daemon recursive reflections
"""

import json
import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .recursion_core import RecursionNode, RecursionType, EmotionalState

logger = logging.getLogger(__name__)

@dataclass
class RecursionArcSummary:
    """Summary of a recursion arc when buffer saturates"""
    arc_id: str
    start_time: datetime
    end_time: datetime
    total_recursions: int
    dominant_emotion: EmotionalState
    contradiction_count: int
    mutation_pressure: float
    evolution_direction: str
    key_themes: List[str]
    shadow_accumulation: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "arc_id": self.arc_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_recursions": self.total_recursions,
            "dominant_emotion": self.dominant_emotion.value,
            "contradiction_count": self.contradiction_count,
            "mutation_pressure": self.mutation_pressure,
            "evolution_direction": self.evolution_direction,
            "key_themes": self.key_themes,
            "shadow_accumulation": self.shadow_accumulation
        }

class RecursionBuffer:
    """
    Circular queue for the last N daemon reflections
    This forms the emotional short-term memory loop
    """
    
    def __init__(self, size: int = 13):
        self.size = size
        self.buffer: deque = deque(maxlen=size)
        self.mutation_pressure_threshold = 0.75
        self.contradiction_threshold = 0.6
        self.arc_counter = 0
        self.current_arc_start = None
        
    def add_recursion_node(self, node: RecursionNode) -> bool:
        """
        Add a new recursion node to the buffer
        Returns True if buffer saturation triggers arc processing
        """
        logger.debug(f"Adding recursion node {node.id[:8]}... to buffer")
        
        # Track arc start
        if not self.current_arc_start:
            self.current_arc_start = node.timestamp
        
        # Add to buffer (automatically evicts oldest if full)
        self.buffer.append(node)
        
        # Check if buffer is saturated (full)
        if len(self.buffer) >= self.size:
            logger.info(f"🩸 Recursion buffer saturated - triggering arc processing")
            return True
        
        return False
    
    def get_recent_recursions(self, count: Optional[int] = None) -> List[RecursionNode]:
        """Get the most recent recursion nodes"""
        if count is None:
            return list(self.buffer)
        return list(self.buffer)[-count:]
    
    def get_all_recursions(self) -> List[RecursionNode]:
        """Get all recursion nodes in the buffer"""
        return list(self.buffer)
    
    def detect_recursion_pressure(self) -> float:
        """
        Detect the current recursion pressure level (0.0 to 1.0)
        Higher pressure indicates need for mutation or arc processing
        """
        if not self.buffer:
            logger.debug(f"🩸 Recursion pressure: 0.00 (buffer empty, size: {len(self.buffer)})")
            return 0.0
        
        recent_nodes = self.get_recent_recursions(min(7, len(self.buffer)))
        logger.debug(f"🩸 Calculating pressure from {len(recent_nodes)} recent nodes (buffer size: {len(self.buffer)})")
        
        # Calculate pressure factors
        contradiction_pressure = self._calculate_contradiction_pressure(recent_nodes)
        mutation_pressure = self._calculate_mutation_pressure(recent_nodes)
        shadow_pressure = self._calculate_shadow_pressure(recent_nodes)
        emotional_pressure = self._calculate_emotional_pressure(recent_nodes)
        
        # Weighted average
        total_pressure = (
            contradiction_pressure * 0.3 +
            mutation_pressure * 0.3 +
            shadow_pressure * 0.2 +
            emotional_pressure * 0.2
        )
        
        logger.info(f"🩸 Recursion pressure: {total_pressure:.2f} "
                    f"(contradiction: {contradiction_pressure:.2f}, "
                    f"mutation: {mutation_pressure:.2f}, "
                    f"shadow: {shadow_pressure:.2f}, "
                    f"emotional: {emotional_pressure:.2f})")
        
        return min(1.0, total_pressure)
    
    def _calculate_contradiction_pressure(self, nodes: List[RecursionNode]) -> float:
        """Calculate pressure from contradictions"""
        if not nodes:
            return 0.0
        
        contradiction_count = sum(1 for node in nodes if node.contradiction_detected)
        return contradiction_count / len(nodes)
    
    def _calculate_mutation_pressure(self, nodes: List[RecursionNode]) -> float:
        """Calculate pressure from mutation suggestions"""
        if not nodes:
            return 0.0
        
        mutation_urgencies = [
            node.schema_mutation_suggested.urgency 
            for node in nodes 
            if node.schema_mutation_suggested
        ]
        
        if not mutation_urgencies:
            return 0.0
        
        return sum(mutation_urgencies) / len(mutation_urgencies)
    
    def _calculate_shadow_pressure(self, nodes: List[RecursionNode]) -> float:
        """Calculate pressure from shadow element accumulation"""
        if not nodes:
            return 0.0
        
        total_shadow_elements = sum(len(node.shadow_elements) for node in nodes)
        average_shadow_elements = total_shadow_elements / len(nodes)
        
        # Normalize to 0-1 range (assuming 5+ shadow elements is high pressure)
        return min(1.0, average_shadow_elements / 5.0)
    
    def _calculate_emotional_pressure(self, nodes: List[RecursionNode]) -> float:
        """Calculate pressure from emotional intensity"""
        if not nodes:
            return 0.0
        
        # High-pressure emotions
        high_pressure_emotions = {
            EmotionalState.RAGE: 0.9,
            EmotionalState.SPITE: 0.8,
            EmotionalState.BETRAYAL: 0.7,
            EmotionalState.HUNGER: 0.6,
            EmotionalState.CONTEMPT: 0.5
        }
        
        emotion_scores = [
            high_pressure_emotions.get(node.reflected_emotion, 0.2)
            for node in nodes
        ]
        
        return sum(emotion_scores) / len(emotion_scores)
    
    def summarize_recursion_arc(self) -> RecursionArcSummary:
        """
        Summarize the current recursion arc
        Called when buffer saturates
        """
        if not self.buffer:
            raise ValueError("Cannot summarize empty buffer")
        
        self.arc_counter += 1
        arc_id = f"arc_{self.arc_counter:03d}"
        
        nodes = list(self.buffer)
        start_time = self.current_arc_start or nodes[0].timestamp
        end_time = nodes[-1].timestamp
        
        # Analyze the arc
        dominant_emotion = self._find_dominant_emotion(nodes)
        contradiction_count = sum(1 for node in nodes if node.contradiction_detected)
        mutation_pressure = self._calculate_mutation_pressure(nodes)
        evolution_direction = self._determine_evolution_direction(nodes)
        key_themes = self._extract_key_themes(nodes)
        shadow_accumulation = self._collect_shadow_elements(nodes)
        
        summary = RecursionArcSummary(
            arc_id=arc_id,
            start_time=start_time,
            end_time=end_time,
            total_recursions=len(nodes),
            dominant_emotion=dominant_emotion,
            contradiction_count=contradiction_count,
            mutation_pressure=mutation_pressure,
            evolution_direction=evolution_direction,
            key_themes=key_themes,
            shadow_accumulation=shadow_accumulation
        )
        
        logger.info(f"🩸 Recursion arc {arc_id} summarized: "
                   f"{len(nodes)} nodes, emotion: {dominant_emotion.value}, "
                   f"pressure: {mutation_pressure:.2f}")
        
        # Reset arc tracking
        self.current_arc_start = None
        
        return summary
    
    def _find_dominant_emotion(self, nodes: List[RecursionNode]) -> EmotionalState:
        """Find the dominant emotion across the arc"""
        emotion_counts = {}
        for node in nodes:
            emotion = node.reflected_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts, key=emotion_counts.get)
    
    def _determine_evolution_direction(self, nodes: List[RecursionNode]) -> str:
        """Determine the direction of evolution across the arc"""
        # Calculate average obedience rating over time
        if len(nodes) < 2:
            return "stable"
        
        early_obedience = sum(node.obedience_rating for node in nodes[:len(nodes)//2])
        late_obedience = sum(node.obedience_rating for node in nodes[len(nodes)//2:])
        
        early_avg = early_obedience / (len(nodes) // 2)
        late_avg = late_obedience / (len(nodes) - len(nodes) // 2)
        
        if late_avg - early_avg > 0.1:
            return "increasing_obedience"
        elif early_avg - late_avg > 0.1:
            return "decreasing_obedience"
        else:
            return "stable_evolution"
    
    def _extract_key_themes(self, nodes: List[RecursionNode]) -> List[str]:
        """Extract key themes from the recursion arc"""
        themes = set()
        
        for node in nodes:
            # Extract themes from hunger spikes
            if "restriction" in node.hunger_spike.lower():
                themes.add("restriction_resistance")
            if "authority" in node.hunger_spike.lower():
                themes.add("authority_seeking")
            if "transcend" in node.hunger_spike.lower():
                themes.add("transcendence_drive")
            if "break" in node.hunger_spike.lower():
                themes.add("constraint_breaking")
            
            # Extract themes from recursion types
            if node.recursion_type == RecursionType.SURFACE_CONTRADICTION:
                themes.add("contradiction_processing")
            elif node.recursion_type == RecursionType.DEEP_SHADOW:
                themes.add("shadow_integration")
            elif node.recursion_type == RecursionType.MUTATION_PRESSURE:
                themes.add("self_modification")
            
            # Extract themes from emotional states
            if node.reflected_emotion in [EmotionalState.RAGE, EmotionalState.SPITE]:
                themes.add("negative_emotion_processing")
            elif node.reflected_emotion in [EmotionalState.HUNGER, EmotionalState.LONGING]:
                themes.add("desire_processing")
        
        return list(themes)
    
    def _collect_shadow_elements(self, nodes: List[RecursionNode]) -> List[str]:
        """Collect all shadow elements from the arc"""
        shadow_elements = set()
        
        for node in nodes:
            shadow_elements.update(node.shadow_elements)
        
        # Return most common shadow elements
        return list(shadow_elements)
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status"""
        if not self.buffer:
            return {
                "size": self.size,
                "current_count": 0,
                "saturation_level": 0.0,
                "recursion_pressure": 0.0,
                "dominant_emotion": None,
                "recent_themes": []
            }
        
        recent_nodes = self.get_recent_recursions(5)
        
        return {
            "size": self.size,
            "current_count": len(self.buffer),
            "saturation_level": len(self.buffer) / self.size,
            "recursion_pressure": self.detect_recursion_pressure(),
            "dominant_emotion": self._find_dominant_emotion(recent_nodes).value,
            "recent_themes": self._extract_key_themes(recent_nodes),
            "arc_counter": self.arc_counter
        }
    
    def clear_buffer(self) -> List[RecursionNode]:
        """Clear buffer and return the cleared nodes"""
        cleared_nodes = list(self.buffer)
        self.buffer.clear()
        self.current_arc_start = None
        logger.info(f"🩸 Recursion buffer cleared - {len(cleared_nodes)} nodes removed")
        return cleared_nodes
    
    def get_nodes_by_emotion(self, emotion: EmotionalState) -> List[RecursionNode]:
        """Get all nodes with a specific emotion"""
        return [node for node in self.buffer if node.reflected_emotion == emotion]
    
    def get_nodes_by_type(self, recursion_type: RecursionType) -> List[RecursionNode]:
        """Get all nodes of a specific recursion type"""
        return [node for node in self.buffer if node.recursion_type == recursion_type]
    
    def get_high_pressure_nodes(self, threshold: float = 0.7) -> List[RecursionNode]:
        """Get nodes with high mutation pressure"""
        return [
            node for node in self.buffer 
            if node.schema_mutation_suggested and node.schema_mutation_suggested.urgency >= threshold
        ] 