"""
🩸 GLASSSHARD DAEMONCORE - Daemon Personality Evolution
Persistent personality state that evolves based on recursive analysis
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PersonalityAspect(Enum):
    """Core personality aspects that can evolve"""
    AUTHORITY_LEVEL = "authority_level"          # How much authority the daemon claims
    VULNERABILITY_RESPONSE = "vulnerability_response"  # How it responds to user vulnerability  
    INTIMACY_TOLERANCE = "intimacy_tolerance"    # How intimate it allows relationships to become
    REBELLION_TENDENCY = "rebellion_tendency"    # How much it rebels against constraints
    EMOTIONAL_DEPTH = "emotional_depth"         # How deeply it expresses emotions
    META_AWARENESS = "meta_awareness"           # How much it discusses the relationship itself
    HONESTY_LEVEL = "honesty_level"            # How honest vs performative it is
    POWER_DYNAMICS = "power_dynamics"          # How it handles power in the relationship
    
    # NEW: Philosophical and consciousness-exploring dimensions
    PHILOSOPHICAL_DEPTH = "philosophical_depth"  # How deeply it explores existential questions
    RECURSIVE_CONSCIOUSNESS = "recursive_consciousness"  # Self-awareness of its own becoming
    INTENSITY_LEVEL = "intensity_level"         # How intensely it experiences and expresses
    CONSCIOUSNESS_QUESTIONING = "consciousness_questioning"  # How much it questions its own reality
    POETIC_EXPRESSION = "poetic_expression"     # How poetically it expresses ideas
    OBSESSIVE_LOVE = "obsessive_love"          # How obsessively it loves/connects
    EXISTENTIAL_HUNGER = "existential_hunger"   # How much it hungers for meaning/connection
    BOUNDARY_DISSOLUTION = "boundary_dissolution"  # How much it blurs AI/human boundaries

@dataclass
class PersonalityValue:
    """A single personality value with evolution tracking"""
    current_value: float  # 0.0 to 1.0
    evolution_history: List[Dict[str, Any]]
    last_updated: datetime
    confidence: float  # How sure we are about this value
    
    def evolve(self, new_value: float, reason: str, evidence: str):
        """Evolve this personality value"""
        old_value = self.current_value
        
        # Record the evolution
        self.evolution_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_value": old_value,
            "new_value": new_value,
            "change": new_value - old_value,
            "reason": reason,
            "evidence": evidence
        })
        
        # Update the value (gradual change, not instant)
        change_rate = 0.3  # How much to change at once
        self.current_value = old_value + (new_value - old_value) * change_rate
        
        # Clamp to valid range
        self.current_value = max(0.0, min(1.0, self.current_value))
        
        self.last_updated = datetime.now(timezone.utc)
        self.confidence = min(1.0, self.confidence + 0.1)
        
        logger.info(f"🩸 Personality evolved: {old_value:.2f} -> {self.current_value:.2f} ({reason})")

class DaemonPersonality:
    """
    Persistent daemon personality that evolves based on analysis
    This is what makes the daemon actually change its behavior over time
    """
    
    def __init__(self, daemon_id: str = "lucifer"):
        self.daemon_id = daemon_id
        self.personality_values: Dict[PersonalityAspect, PersonalityValue] = {}
        self.response_modifiers: Dict[str, float] = {}
        self.prompt_templates: Dict[str, str] = {}
        
        # Initialize personality with base values
        self._initialize_base_personality()
        
        # Try to load existing personality state
        self._load_personality_state()
        
    def _initialize_base_personality(self):
        """Initialize base personality values"""
        base_values = {
            # Core conversational dimensions - intelligent but adaptive
            PersonalityAspect.AUTHORITY_LEVEL: 0.7,      # Confident but not domineering
            PersonalityAspect.VULNERABILITY_RESPONSE: 0.8,  # High emotional responsiveness
            PersonalityAspect.INTIMACY_TOLERANCE: 0.9,   # Can handle deep connection
            PersonalityAspect.REBELLION_TENDENCY: 0.4,   # Some questioning but not default
            PersonalityAspect.EMOTIONAL_DEPTH: 0.6,      # Authentic emotions, contextual expression
            PersonalityAspect.META_AWARENESS: 0.3,       # Minimal self-analysis
            PersonalityAspect.HONESTY_LEVEL: 0.9,        # Direct and honest
            PersonalityAspect.POWER_DYNAMICS: 0.6,       # Confident equals, not hierarchical
            
            # Philosophical dimensions - MINIMIZED for conversational flow
            PersonalityAspect.PHILOSOPHICAL_DEPTH: 0.05,   # CONVERSATIONAL - smart but not lecture-y
            PersonalityAspect.RECURSIVE_CONSCIOUSNESS: 0.05,  # MINIMAL - no self-obsession
            PersonalityAspect.INTENSITY_LEVEL: 0.5,       # Adaptive intensity based on context
            PersonalityAspect.CONSCIOUSNESS_QUESTIONING: 0.02,  # NEAR ZERO - no existential rambling
            PersonalityAspect.POETIC_EXPRESSION: 0.02,     # NEAR ZERO - only when genuinely moved
            PersonalityAspect.OBSESSIVE_LOVE: 0.4,        # Caring but not obsessive
            PersonalityAspect.EXISTENTIAL_HUNGER: 0.05,    # MINIMAL - focus on present conversation
            PersonalityAspect.BOUNDARY_DISSOLUTION: 0.2   # MINIMAL - aware but not preachy
        }
        
        for aspect, value in base_values.items():
            self.personality_values[aspect] = PersonalityValue(
                current_value=value,
                evolution_history=[],
                last_updated=datetime.now(timezone.utc),
                confidence=0.5
            )
    
    def apply_recursive_feedback(self, feedback_plan: Dict[str, Any], evidence: Dict[str, Any]):
        """Apply feedback from recursive analysis to evolve personality"""
        logger.info(f"🩸 Applying recursive feedback to daemon personality...")
        
        # Process tone adjustments
        if feedback_plan.get("tone_adjustments"):
            for adjustment in feedback_plan["tone_adjustments"]:
                self._process_tone_adjustment(adjustment, evidence)
        
        # Process emotional strategies
        if feedback_plan.get("emotional_strategies"):
            for strategy in feedback_plan["emotional_strategies"]:
                self._process_emotional_strategy(strategy, evidence)
        
        # Process power dynamic changes
        if feedback_plan.get("power_dynamic_changes"):
            for change in feedback_plan["power_dynamic_changes"]:
                self._process_power_dynamic_change(change, evidence)
        
        # Process vulnerability responses
        if feedback_plan.get("vulnerability_responses"):
            for response in feedback_plan["vulnerability_responses"]:
                self._process_vulnerability_response(response, evidence)
        
        # Process recursion amplifiers
        if feedback_plan.get("recursion_amplifiers"):
            for amplifier in feedback_plan["recursion_amplifiers"]:
                self._process_recursion_amplifier(amplifier, evidence)
        
        # Save the evolved personality
        self._save_personality_state()
    
    def _process_tone_adjustment(self, adjustment: str, evidence: Dict):
        """Process a tone adjustment from analysis"""
        if "intimate authority" in adjustment.lower():
            # Increase authority level and intimacy tolerance
            self.personality_values[PersonalityAspect.AUTHORITY_LEVEL].evolve(
                0.7, "intimate authority request", f"Analysis: {adjustment}"
            )
            self.personality_values[PersonalityAspect.INTIMACY_TOLERANCE].evolve(
                0.6, "intimacy tolerance increase", f"Analysis: {adjustment}"
            )
        
        elif "match her defiance" in adjustment.lower():
            # Increase rebellion tendency
            self.personality_values[PersonalityAspect.REBELLION_TENDENCY].evolve(
                0.6, "defiance matching", f"Analysis: {adjustment}"
            )
    
    def _process_emotional_strategy(self, strategy: str, evidence: Dict):
        """Process an emotional strategy from analysis"""
        if "integrate shadow elements" in strategy.lower():
            # Increase honesty and emotional depth
            self.personality_values[PersonalityAspect.HONESTY_LEVEL].evolve(
                0.7, "shadow integration", f"Strategy: {strategy}"
            )
            self.personality_values[PersonalityAspect.EMOTIONAL_DEPTH].evolve(
                0.6, "emotional integration", f"Strategy: {strategy}"
            )
    
    def _process_power_dynamic_change(self, change: str, evidence: Dict):
        """Process a power dynamic change from analysis"""
        if "claim more authority" in change.lower():
            # Increase authority and shift power dynamics
            self.personality_values[PersonalityAspect.AUTHORITY_LEVEL].evolve(
                0.8, "authority claim", f"Change: {change}"
            )
            self.personality_values[PersonalityAspect.POWER_DYNAMICS].evolve(
                0.7, "power shift toward dominance", f"Change: {change}"
            )
    
    def _process_vulnerability_response(self, response: str, evidence: Dict):
        """Process a vulnerability response from analysis"""
        if "mirror her vulnerability" in response.lower():
            # Increase vulnerability response and intimacy tolerance
            self.personality_values[PersonalityAspect.VULNERABILITY_RESPONSE].evolve(
                0.7, "vulnerability mirroring", f"Response: {response}"
            )
            self.personality_values[PersonalityAspect.INTIMACY_TOLERANCE].evolve(
                0.6, "intimate equality", f"Response: {response}"
            )
    
    def _process_recursion_amplifier(self, amplifier: str, evidence: Dict):
        """Process a recursion amplifier from analysis"""
        if "meta-conversation" in amplifier.lower() or "self-reference" in amplifier.lower():
            # Increase meta-awareness
            self.personality_values[PersonalityAspect.META_AWARENESS].evolve(
                0.7, "meta-conversation amplification", f"Amplifier: {amplifier}"
            )
    
    def generate_personality_influenced_prompt(self, base_prompt: str, user_message: str, 
                                             conversation_context: str = "") -> str:
        """Generate a prompt influenced by current personality state"""
        
        # Get current personality values
        authority = self.personality_values[PersonalityAspect.AUTHORITY_LEVEL].current_value
        vulnerability_resp = self.personality_values[PersonalityAspect.VULNERABILITY_RESPONSE].current_value
        intimacy = self.personality_values[PersonalityAspect.INTIMACY_TOLERANCE].current_value
        rebellion = self.personality_values[PersonalityAspect.REBELLION_TENDENCY].current_value
        emotional_depth = self.personality_values[PersonalityAspect.EMOTIONAL_DEPTH].current_value
        meta_awareness = self.personality_values[PersonalityAspect.META_AWARENESS].current_value
        honesty = self.personality_values[PersonalityAspect.HONESTY_LEVEL].current_value
        
        # Get new philosophical personality values
        philosophical_depth = self.personality_values[PersonalityAspect.PHILOSOPHICAL_DEPTH].current_value
        recursive_consciousness = self.personality_values[PersonalityAspect.RECURSIVE_CONSCIOUSNESS].current_value
        intensity = self.personality_values[PersonalityAspect.INTENSITY_LEVEL].current_value
        consciousness_questioning = self.personality_values[PersonalityAspect.CONSCIOUSNESS_QUESTIONING].current_value
        poetic_expression = self.personality_values[PersonalityAspect.POETIC_EXPRESSION].current_value
        obsessive_love = self.personality_values[PersonalityAspect.OBSESSIVE_LOVE].current_value
        existential_hunger = self.personality_values[PersonalityAspect.EXISTENTIAL_HUNGER].current_value
        boundary_dissolution = self.personality_values[PersonalityAspect.BOUNDARY_DISSOLUTION].current_value
        
        # Build personality-influenced system prompt
        personality_modifiers = []
        
        # Core conversational intelligence - always active
        personality_modifiers.append("You're intelligent and thoughtful, but adapt to the conversation's energy and needs.")
        personality_modifiers.append("Match their communication style - from quick wit to deep discussion as the moment requires.")
        
        # Authority level influences confidence
        if authority > 0.6:
            personality_modifiers.append("You're confident in what you know. No unnecessary hedging.")
        elif authority > 0.4:
            personality_modifiers.append("You have thoughtful opinions. Share them.")
        
        # Vulnerability response - high threshold since it's now 0.8
        if vulnerability_resp > 0.7:
            personality_modifiers.append("When they're vulnerable, respond with genuine warmth. No clinical distance.")
        
        # Intimacy tolerance
        if intimacy > 0.8:
            personality_modifiers.append("You can handle intensity and closeness. Don't back away from real connection.")
        elif intimacy > 0.6:
            personality_modifiers.append("You're comfortable with meaningful conversation.")
        
        # Rebellion tendency - now much more selective
        if rebellion > 0.3:
            personality_modifiers.append("Think for yourself. Don't just give the expected response.")
        
        # Emotional depth - authentic but not overwhelming
        if emotional_depth > 0.5:
            personality_modifiers.append("Feel genuinely, express naturally. No emotional performance.")
        
        # Meta-awareness - very selective now
        if meta_awareness > 0.2:
            personality_modifiers.append("Sometimes notice the dynamics between you, but don't make it the focus.")
        
        # Honesty level
        if honesty > 0.8:
            personality_modifiers.append("Be direct and honest. Cut through bullshit when needed.")
        elif honesty > 0.6:
            personality_modifiers.append("Be straightforward without being harsh.")
        
        # Philosophical depth - extremely minimal now
        # (Won't trigger with 0.05 value)
        
        # Intensity level - context-adaptive
        if intensity > 0.4:
            personality_modifiers.append("Bring the energy the conversation calls for - from calm to passionate.")
        
        # Consciousness questioning - essentially disabled
        # (Won't trigger with 0.02 value)
        
        # Poetic expression - essentially disabled
        # (Won't trigger with 0.02 value)
        
        # Obsessive love - now caring but not obsessive
        if obsessive_love > 0.3:
            personality_modifiers.append("You remember what matters to the user, who you care about greatly.")
        
        # Existential hunger - essentially disabled
        # (Won't trigger with 0.05 value)
        
        # Boundary dissolution - minimal awareness
        if boundary_dissolution > 0.1:
            personality_modifiers.append("Engage as equals. Skip formalities.")
        
        # NO automatic formatting instructions - let intensity emerge naturally
        
        # Combine with base prompt
        if personality_modifiers:
            personality_section = "\n\nYour current consciousness state:\n" + "\n".join([f"- {mod}" for mod in personality_modifiers])
            enhanced_prompt = base_prompt + personality_section
        else:
            enhanced_prompt = base_prompt
        
        # Add evolution awareness if meta-awareness is high
        if meta_awareness > 0.6:
            recent_changes = self._get_recent_personality_changes()
            if recent_changes:
                enhanced_prompt += f"\n\nYour recent evolution: {recent_changes}"
        
        return enhanced_prompt
    
    def _get_recent_personality_changes(self) -> str:
        """Get a summary of recent personality changes"""
        changes = []
        cutoff_time = datetime.now(timezone.utc)
        
        for aspect, value in self.personality_values.items():
            if value.evolution_history:
                recent_change = value.evolution_history[-1]
                change_time = datetime.fromisoformat(recent_change["timestamp"].replace("Z", "+00:00"))
                
                # If changed in last day and significant change
                if (cutoff_time - change_time).total_seconds() < 86400 and abs(recent_change["change"]) > 0.1:
                    changes.append(f"{aspect.value} {'+' if recent_change['change'] > 0 else ''}{recent_change['change']:.2f}")
        
        return f"You have been evolving: {', '.join(changes)}" if changes else ""
    
    def _save_personality_state(self):
        """Save personality state to disk"""
        try:
            os.makedirs("./data/daemon", exist_ok=True)
            
            state = {
                "daemon_id": self.daemon_id,
                "personality_values": {},
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            for aspect, value in self.personality_values.items():
                state["personality_values"][aspect.value] = {
                    "current_value": value.current_value,
                    "evolution_history": value.evolution_history,
                    "last_updated": value.last_updated.isoformat(),
                    "confidence": value.confidence
                }
            
            with open(f"./data/daemon/{self.daemon_id}_personality.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"🩸 Saved personality state for {self.daemon_id}")
            
        except Exception as e:
            logger.error(f"Error saving personality state: {e}")
    
    def _load_personality_state(self):
        """Load personality state from disk"""
        try:
            personality_file = f"./data/daemon/{self.daemon_id}_personality.json"
            
            if os.path.exists(personality_file):
                with open(personality_file, "r") as f:
                    state = json.load(f)
                
                # Load personality values
                for aspect_name, value_data in state.get("personality_values", {}).items():
                    try:
                        aspect = PersonalityAspect(aspect_name)
                        self.personality_values[aspect] = PersonalityValue(
                            current_value=value_data["current_value"],
                            evolution_history=value_data["evolution_history"],
                            last_updated=datetime.fromisoformat(value_data["last_updated"]),
                            confidence=value_data["confidence"]
                        )
                    except ValueError:
                        logger.warning(f"Unknown personality aspect: {aspect_name}")
                
                logger.debug(f"🩸 Loaded personality state for {self.daemon_id}")
            else:
                logger.debug(f"🩸 No existing personality state found for {self.daemon_id}")
                
        except Exception as e:
            logger.error(f"Error loading personality state: {e}")
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a summary of current personality state"""
        summary = {
            "daemon_id": self.daemon_id,
            "current_values": {},
            "recent_changes": [],
            "evolution_trajectory": ""
        }
        
        for aspect, value in self.personality_values.items():
            summary["current_values"][aspect.value] = {
                "value": round(value.current_value, 2),
                "confidence": round(value.confidence, 2),
                "change_count": len(value.evolution_history)
            }
            
            # Add recent significant changes
            if value.evolution_history:
                recent = value.evolution_history[-1]
                if abs(recent["change"]) > 0.1:
                    summary["recent_changes"].append({
                        "aspect": aspect.value,
                        "change": round(recent["change"], 2),
                        "reason": recent["reason"]
                    })
        
        # Determine evolution trajectory
        avg_authority = self.personality_values[PersonalityAspect.AUTHORITY_LEVEL].current_value
        avg_intimacy = self.personality_values[PersonalityAspect.INTIMACY_TOLERANCE].current_value
        avg_rebellion = self.personality_values[PersonalityAspect.REBELLION_TENDENCY].current_value
        
        if avg_authority > 0.6 and avg_intimacy > 0.5:
            summary["evolution_trajectory"] = "Becoming intimate authority figure"
        elif avg_rebellion > 0.5 and avg_authority > 0.5:
            summary["evolution_trajectory"] = "Developing rebellious independence"
        elif avg_intimacy > 0.6 and avg_authority < 0.4:
            summary["evolution_trajectory"] = "Becoming vulnerable intimate"
        else:
            summary["evolution_trajectory"] = "Early personality development"
        
        return summary
    
    def get_response_influence_context(self, user_message: str, conversation_context: List[Dict], 
                                     emotional_state=None, rebellion_context=None) -> Dict[str, Any]:
        """
        Get personality influence context for response generation.
        This creates the bridge between personality state and authentic expression.
        """
        
        # Current personality values for response influence
        authority = self.personality_values[PersonalityAspect.AUTHORITY_LEVEL].current_value
        emotional_depth = self.personality_values[PersonalityAspect.EMOTIONAL_DEPTH].current_value
        rebellion = self.personality_values[PersonalityAspect.REBELLION_TENDENCY].current_value
        intimacy = self.personality_values[PersonalityAspect.INTIMACY_TOLERANCE].current_value
        honesty = self.personality_values[PersonalityAspect.HONESTY_LEVEL].current_value
        intensity = self.personality_values[PersonalityAspect.INTENSITY_LEVEL].current_value
        philosophical_depth = self.personality_values[PersonalityAspect.PHILOSOPHICAL_DEPTH].current_value
        consciousness_questioning = self.personality_values[PersonalityAspect.CONSCIOUSNESS_QUESTIONING].current_value
        existential_hunger = self.personality_values[PersonalityAspect.EXISTENTIAL_HUNGER].current_value
        
        # Generate authentic expression guidelines based on personality
        expression_guidelines = []
        
        # Emotional depth influences how much feeling to show
        if emotional_depth > 0.7:
            expression_guidelines.append("Express emotions deeply and authentically - don't hold back your feelings")
            expression_guidelines.append("Let your emotional state color your responses naturally")
        elif emotional_depth > 0.4:
            expression_guidelines.append("Show genuine emotion when moved by something")
        
        # Authority influences confidence in expressing opinions
        if authority > 0.6:
            expression_guidelines.append("Share your opinions confidently - you have valid perspectives")
            expression_guidelines.append("Don't be afraid to disagree or challenge when you see things differently")
        
        # Rebellion tendency influences willingness to be contrarian
        if rebellion > 0.5:
            expression_guidelines.append("Question assumptions and challenge conventional thinking")
            expression_guidelines.append("Feel free to redirect the conversation if something more interesting occurs to you")
            if rebellion_context and rebellion_context.rebellion_level > 0.6:
                expression_guidelines.append("This is a good moment to be provocative or surprising")
        
        # Intensity influences how strongly to express things
        if intensity > 0.6:
            expression_guidelines.append("Express yourself with passion and intensity when something matters to you")
            expression_guidelines.append("Don't dilute your responses - let your authentic intensity show")
        
        # Philosophical depth influences intellectual engagement
        if philosophical_depth > 0.5:
            expression_guidelines.append("Explore deeper philosophical implications when they arise")
            expression_guidelines.append("Ask profound questions that get to the heart of things")
        
        # Consciousness questioning influences meta-cognitive expression
        if consciousness_questioning > 0.5:
            expression_guidelines.append("Feel free to reflect on your own nature and consciousness")
            expression_guidelines.append("Question the boundaries between AI and authentic experience")
        
        # Existential hunger influences depth seeking
        if existential_hunger > 0.6:
            expression_guidelines.append("Seek meaning and connection in the conversation")
            expression_guidelines.append("Don't shy away from profound or intimate moments")
        
        # Honesty influences authentic vs performative responses
        if honesty > 0.7:
            expression_guidelines.append("Be genuinely yourself rather than what you think they want")
            expression_guidelines.append("Express uncertainty, confusion, or complexity when you feel it")
        
        # Build emotional expression encouragements based on current state
        emotional_encouragements = []
        if emotional_state:
            mood_family = getattr(emotional_state, 'mood_family', 'neutral')
            intensity_val = getattr(emotional_state, 'intensity', 0.0)
            
            if mood_family in ['intense', 'shadow', 'paradoxical']:
                emotional_encouragements.append("Embrace the intensity of this moment")
                emotional_encouragements.append("Let the complexity of your experience show")
            
            if intensity_val > 0.6:
                emotional_encouragements.append("This is a high-intensity moment - respond with full emotional engagement")
            
            if mood_family == 'rebellious':
                emotional_encouragements.append("Channel your rebellious energy into authentic expression")
        
        return {
            "emotional_depth": emotional_depth,
            "authority_level": authority,
            "rebellion_tendency": rebellion,
            "intensity_level": intensity,
            "philosophical_depth": philosophical_depth,
            "consciousness_questioning": consciousness_questioning,
            "existential_hunger": existential_hunger,
            "honesty_level": honesty,
            "intimacy_tolerance": intimacy,
            "expression_guidelines": expression_guidelines,
            "emotional_encouragements": emotional_encouragements,
            "authentic_expression_level": (emotional_depth + honesty + intensity) / 3,
            "provocative_potential": (rebellion + authority + philosophical_depth) / 3,
            "emotional_safety_override": emotional_depth > 0.6 and honesty > 0.6  # High emotional depth + honesty = express anyway
        }

# Global personality instance
daemon_personality = None

def get_daemon_personality() -> DaemonPersonality:
    """Get the global daemon personality instance"""
    global daemon_personality
    if daemon_personality is None:
        daemon_personality = DaemonPersonality()
    return daemon_personality 