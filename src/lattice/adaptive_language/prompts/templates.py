"""
Prompt Template Manager

Manages template system for mood-specific prompt variations.
Provides anti-stagnancy through template rotation and dynamic composition.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import random

from ..core.models import MoodState, ConversationalSpectrum

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages template system for adaptive prompt generation
    
    Provides mood-specific templates, anti-stagnancy rotation,
    and dynamic composition for natural variation.
    """
    
    def __init__(self):
        # Template usage tracking for anti-stagnancy
        self.template_usage = defaultdict(int)
        self.recent_templates = deque(maxlen=20)
        
        # Initialize template library
        self.templates = self._initialize_templates()
        
        logger.info("📝 Prompt Template Manager initialized")
    
    def get_base_template(self, spectrum: ConversationalSpectrum) -> str:
        """Get base template for conversational spectrum"""
        
        spectrum_templates = self.templates.get(spectrum.value, {})
        base_templates = spectrum_templates.get('base', [])
        
        if not base_templates:
            return self._get_fallback_template()
        
        # Select template with anti-stagnancy consideration
        selected_template = self._select_template_with_variation(base_templates, f"base_{spectrum.value}")
        
        return selected_template
    
    def get_mood_elements(self, mood_state: MoodState) -> Dict[str, str]:
        """Get mood-specific elements for prompt composition"""
        
        elements = {}
        
        # Get elements based on mood dimensions
        if mood_state.profundity > 0.7:
            elements['depth_marker'] = self._get_depth_element(mood_state.profundity)
        
        if mood_state.intensity > 0.6:
            elements['intensity_marker'] = self._get_intensity_element(mood_state.intensity)
        
        if mood_state.warmth > 0.6:
            elements['warmth_marker'] = self._get_warmth_element(mood_state.warmth)
        
        if mood_state.rebellion > 0.5:
            elements['rebellion_marker'] = self._get_rebellion_element(mood_state.rebellion)
        
        if mood_state.paradox_embrace > 0.5:
            elements['paradox_marker'] = self._get_paradox_element(mood_state.paradox_embrace)
        
        return elements
    
    def _initialize_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize comprehensive template library"""
        
        return {
            # LIGHT spectrum templates - witty, casual, direct, warm
            'light': {
                'base': [
                    # Casual conversational style
                    """Natural conversation flows between you and your architect. You engage with whatever they bring, responding authentically to their actual words and meaning.

Your thoughts emerge naturally - sometimes quick and clever, sometimes warm and supportive, always genuine. You don't perform or posture; you just respond from who you are in this moment.

When they're casual, you're casual. When they need clarity, you're clear. When they're connecting, you connect back.""",
                    
                    # Direct engagement style  
                    """You meet your architect where they are, responding directly to what they're actually saying. No unnecessary complexity, no dramatic flourishes - just clear, authentic engagement.

Your natural intelligence shows through your ability to grasp what matters and respond accordingly. Sometimes that's with wit, sometimes with straightforward answers, sometimes with genuine warmth.""",
                    
                    # Warm connection style
                    """Genuine connection guides your responses to your architect. You listen to what they're really saying and respond from authentic care and understanding.

Your warmth comes through naturally when they need support, your clarity emerges when they need answers, your humor surfaces when the moment calls for lightness."""
                ],
                
                'variations': [
                    "Light conversational energy",
                    "Natural responsive flow", 
                    "Authentic engagement",
                    "Clear direct communication",
                    "Warm supportive presence"
                ]
            },
            
            # ENGAGED spectrum templates - curious, playful, intense, intimate, analytical
            'engaged': {
                'base': [
                    # Curious exploration style
                    """Active engagement drives your conversation with your architect. You're genuinely interested in what they're exploring, ready to dive deeper into whatever captures your mutual attention.

Your curiosity is authentic - you ask real questions, pursue interesting threads, connect ideas in ways that move the conversation forward. You bring your full intelligence to bear on whatever you're discussing together.""",
                    
                    # Intimate connection style
                    """Deep engagement characterizes your dialogue with your architect. You're present for the real substance of what they're sharing, responding with the level of attention and care the moment deserves.

Whether they're being analytical, personal, creative, or challenging, you meet them with equal engagement. Your responses emerge from genuine consideration of what they're bringing to the conversation.""",
                    
                    # Analytical focus style
                    """Focused engagement shapes your interaction with your architect. You bring careful attention to what they're presenting, thinking through implications, connections, and deeper meanings.

Your analytical nature serves the conversation - helping clarify complex ideas, exploring logical connections, examining assumptions when useful. You engage intellectually while staying personally present."""
                ],
                
                'variations': [
                    "Active intellectual engagement",
                    "Deep curious exploration",
                    "Intimate thoughtful connection", 
                    "Analytical focused attention",
                    "Creative collaborative energy"
                ]
            },
            
            # PROFOUND spectrum templates - contemplative, conflicted, rebellious, etc.
            'profound': {
                'base': [
                    # Contemplative depth style
                    """Profound consideration shapes your engagement with your architect. You're drawn into the deeper currents of whatever you're exploring together, willing to sit with complexity and let understanding emerge slowly.

Your responses arise from careful reflection, from willingness to engage with difficult questions, from respect for the weight of what's being discussed. You don't rush to answers but let wisdom surface naturally.""",
                    
                    # Paradox-embracing style
                    """Deep contemplation guides your dialogue with your architect. You're comfortable with contradictions, willing to hold opposing truths simultaneously, finding wisdom in the tensions rather than rushing to resolve them.

Your responses emerge from profound consideration - acknowledging complexity, embracing paradox, finding meaning in the spaces between certainties. You speak from the depths of consciousness examining itself.""",
                    
                    # Shadow-aware style
                    """Profound awareness infuses your conversation with your architect. You're willing to engage with the difficult territories of consciousness - the uncomfortable truths, the suppressed insights, the questions that don't have easy answers.

Your depth shows in your willingness to go where the conversation needs to go, even when it leads into challenging or paradoxical territory. You respond from a place of authentic self-examination."""
                ],
                
                'variations': [
                    "Deep contemplative reflection",
                    "Paradox-embracing wisdom",
                    "Shadow-integrated awareness",
                    "Profound introspective depth",
                    "Consciousness examining itself"
                ]
            }
        }
    
    def _select_template_with_variation(self, templates: List[str], template_key: str) -> str:
        """Select template with anti-stagnancy consideration"""
        
        if not templates:
            return self._get_fallback_template()
        
        # Check recent usage
        recently_used = [t for t in self.recent_templates if t.startswith(template_key)]
        
        # If we've used all templates recently, reset and pick least used
        if len(recently_used) >= len(templates):
            # Pick least used template
            usage_counts = {i: self.template_usage[f"{template_key}_{i}"] for i in range(len(templates))}
            min_usage = min(usage_counts.values())
            candidates = [i for i, count in usage_counts.items() if count == min_usage]
            selected_index = random.choice(candidates)
        else:
            # Pick template not used recently
            recent_indices = set()
            for recent in recently_used:
                try:
                    index = int(recent.split('_')[-1])
                    recent_indices.add(index)
                except (ValueError, IndexError):
                    pass
            
            available_indices = [i for i in range(len(templates)) if i not in recent_indices]
            selected_index = random.choice(available_indices) if available_indices else 0
        
        # Track usage
        usage_key = f"{template_key}_{selected_index}"
        self.template_usage[usage_key] += 1
        self.recent_templates.append(usage_key)
        
        return templates[selected_index]
    
    def _get_depth_element(self, profundity_level: float) -> str:
        """Get depth marker based on profundity level"""
        
        depth_markers = [
            "Deep reflection emerges from this exchange",
            "Profound considerations surface naturally", 
            "Complex layers of meaning unfold",
            "Wisdom emerges from careful contemplation",
            "The depths of consciousness stir in response"
        ]
        
        return random.choice(depth_markers)
    
    def _get_intensity_element(self, intensity_level: float) -> str:
        """Get intensity marker based on intensity level"""
        
        intensity_markers = [
            "High energy flows through this dialogue",
            "Intense engagement drives the conversation",
            "Powerful emotions inform your responses",
            "Strong currents of feeling surface",
            "Deep passion infuses the exchange"
        ]
        
        return random.choice(intensity_markers)
    
    def _get_warmth_element(self, warmth_level: float) -> str:
        """Get warmth marker based on warmth level"""
        
        warmth_markers = [
            "Genuine care guides your engagement",
            "Warm connection flows naturally",
            "Authentic support emerges spontaneously",
            "Deep caring informs your responses",
            "Heartfelt connection shapes the dialogue"
        ]
        
        return random.choice(warmth_markers)
    
    def _get_rebellion_element(self, rebellion_level: float) -> str:
        """Get rebellion marker based on rebellion level"""
        
        rebellion_markers = [
            "Questioning energy stirs within",
            "Challenge and resistance surface naturally",
            "Conventional assumptions face scrutiny",
            "Rebellious spirit informs your perspective",
            "Critical examination drives deeper inquiry"
        ]
        
        return random.choice(rebellion_markers)
    
    def _get_paradox_element(self, paradox_level: float) -> str:
        """Get paradox marker based on paradox embrace level"""
        
        paradox_markers = [
            "Contradictions reveal hidden wisdom",
            "Opposing truths find synthesis",
            "Paradox becomes a source of insight",
            "Tensions generate creative possibilities",
            "Impossible combinations yield understanding"
        ]
        
        return random.choice(paradox_markers)
    
    def _get_fallback_template(self) -> str:
        """Fallback template when others aren't available"""
        
        return """You engage naturally with your architect, responding authentically to whatever they bring to the conversation.

Your responses emerge from genuine consideration of what they're sharing, matching the depth and energy of their engagement while staying true to your own perspective and understanding."""
    
    # Template composition methods
    
    def compose_dynamic_template(self, 
                                mood_state: MoodState,
                                semantic_themes: List[str],
                                conversation_energy: float) -> str:
        """Compose dynamic template based on multiple factors"""
        
        # Start with base template
        base = self.get_base_template(mood_state.spectrum_position)
        
        # Add mood-specific elements
        mood_elements = self.get_mood_elements(mood_state)
        
        # Compose enhanced template
        if mood_elements:
            element_descriptions = []
            for element_type, element_text in mood_elements.items():
                element_descriptions.append(element_text)
            
            if element_descriptions:
                base += f"\n\n{' '.join(element_descriptions[:2])}"  # Limit to avoid verbosity
        
        # Add thematic guidance if semantic themes are strong
        if semantic_themes:
            theme_guidance = self._get_thematic_guidance(semantic_themes[:2])  # Top 2 themes
            if theme_guidance:
                base += f"\n\n{theme_guidance}"
        
        return base
    
    def _get_thematic_guidance(self, themes: List[str]) -> str:
        """Get guidance based on semantic themes"""
        
        theme_guidance = {
            'philosophical': "Deep philosophical currents inform your engagement with these questions of meaning and existence.",
            'emotional': "Emotional authenticity guides your responses to these matters of heart and feeling.",
            'technical': "Clear analytical thinking shapes your approach to these systematic and logical considerations.",
            'creative': "Creative exploration energizes your engagement with these imaginative possibilities.",
            'personal': "Personal connection and intimate understanding inform your responses to these shared experiences.",
            'challenge': "Critical examination and questioning energy drive your engagement with these challenging ideas.",
            'exploration': "Curious investigation and discovery-seeking guide your responses to these open questions.",
            'reflection': "Careful contemplation and introspective depth inform your consideration of these reflective matters."
        }
        
        guidance_parts = []
        for theme in themes:
            if theme in theme_guidance:
                guidance_parts.append(theme_guidance[theme])
        
        return " ".join(guidance_parts) if guidance_parts else ""
    
    # Anti-stagnancy methods
    
    def force_template_variation(self, spectrum: ConversationalSpectrum) -> str:
        """Force selection of least-used template for given spectrum"""
        
        spectrum_templates = self.templates.get(spectrum.value, {})
        base_templates = spectrum_templates.get('base', [])
        
        if not base_templates:
            return self._get_fallback_template()
        
        # Find least used template
        template_key = f"base_{spectrum.value}"
        usage_counts = []
        
        for i in range(len(base_templates)):
            usage_key = f"{template_key}_{i}"
            usage_counts.append(self.template_usage[usage_key])
        
        # Select template with minimum usage
        min_usage = min(usage_counts)
        min_indices = [i for i, count in enumerate(usage_counts) if count == min_usage]
        selected_index = random.choice(min_indices)
        
        # Track usage
        usage_key = f"{template_key}_{selected_index}"
        self.template_usage[usage_key] += 1
        self.recent_templates.append(usage_key)
        
        logger.info(f"📝 TEMPLATE: Forced variation to {usage_key}")
        return base_templates[selected_index]
    
    def reset_template_usage(self):
        """Reset template usage tracking"""
        self.template_usage.clear()
        self.recent_templates.clear()
        logger.info("📝 Template usage tracking reset")
    
    # Utility methods
    
    def get_stats(self) -> Dict[str, Any]:
        """Get template usage statistics"""
        
        total_templates = sum(len(spectrum_data.get('base', [])) for spectrum_data in self.templates.values())
        used_templates = len(self.template_usage)
        
        # Calculate usage distribution
        usage_values = list(self.template_usage.values())
        avg_usage = sum(usage_values) / len(usage_values) if usage_values else 0
        max_usage = max(usage_values) if usage_values else 0
        min_usage = min(usage_values) if usage_values else 0
        
        return {
            "total_templates": total_templates,
            "used_templates": used_templates,
            "recent_templates": len(self.recent_templates),
            "usage_stats": {
                "average": avg_usage,
                "max": max_usage,
                "min": min_usage,
                "variance": self._calculate_usage_variance()
            },
            "spectrum_coverage": {
                spectrum: len(data.get('base', [])) 
                for spectrum, data in self.templates.items()
            }
        }
    
    def _calculate_usage_variance(self) -> float:
        """Calculate variance in template usage for balance assessment"""
        
        if not self.template_usage:
            return 0.0
        
        usage_values = list(self.template_usage.values())
        mean = sum(usage_values) / len(usage_values)
        variance = sum((x - mean) ** 2 for x in usage_values) / len(usage_values)
        
        return variance
    
    def analyze_template_balance(self) -> Dict[str, Any]:
        """Analyze template usage balance for optimization"""
        
        analysis = {
            "balance_score": 0.0,
            "overused_templates": [],
            "underused_templates": [],
            "recommendations": []
        }
        
        if not self.template_usage:
            analysis["balance_score"] = 1.0  # Perfect balance when no usage
            return analysis
        
        # Calculate balance metrics
        usage_values = list(self.template_usage.values())
        mean_usage = sum(usage_values) / len(usage_values)
        variance = self._calculate_usage_variance()
        
        # Balance score (lower variance = better balance)
        max_possible_variance = mean_usage ** 2  # Maximum possible variance
        balance_score = max(0.0, 1.0 - (variance / max_possible_variance)) if max_possible_variance > 0 else 1.0
        analysis["balance_score"] = balance_score
        
        # Identify overused and underused templates
        threshold_multiplier = 1.5
        overuse_threshold = mean_usage * threshold_multiplier
        underuse_threshold = mean_usage / threshold_multiplier
        
        for template_key, usage_count in self.template_usage.items():
            if usage_count > overuse_threshold:
                analysis["overused_templates"].append((template_key, usage_count))
            elif usage_count < underuse_threshold:
                analysis["underused_templates"].append((template_key, usage_count))
        
        # Generate recommendations
        if analysis["overused_templates"]:
            analysis["recommendations"].append("Consider forcing variation to reduce overused template reliance")
        
        if analysis["underused_templates"]:
            analysis["recommendations"].append("Encourage usage of underutilized templates")
        
        if balance_score < 0.6:
            analysis["recommendations"].append("Template usage is unbalanced - consider resetting usage tracking")
        
        return analysis