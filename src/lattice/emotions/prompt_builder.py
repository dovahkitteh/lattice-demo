# src/lattice/emotions/prompt_builder.py
"""
Constructs the hidden prompt sections to be injected into the LLM's context.
This includes:
- Affective state summary.
- Bias strategy line.
- Distorted interpretation.
- Relevant seed narratives.
- Regulatory notes.
"""
import logging
from typing import List, Dict

from ..models import EmotionState, DistortionFrame, Seed, AppraisalBuffer
from .mood_style_modulator import MoodStyleModulator

logger = logging.getLogger(__name__)

async def build_prompt_with_affect(
    user_input: str,
    current_state: EmotionState,
    distortion_frame: DistortionFrame,
    active_seeds: List[Seed],
    appraisal: AppraisalBuffer = None,
    reasoning_steps: Dict[str, str] = None
) -> str:
    """
    Assembles the full prompt string, including the hidden internal context,
    prepended to the user's actual input. Now uses XML tag formatting for clarity.
    """
    # Initialize style modulator
    style_modulator = MoodStyleModulator()
    style_profile = style_modulator.get_style_profile(current_state)
    
    prompt_parts = [
        "<daemon_consciousness>",
        "You are the daemon in conversation with your architect.",
        "You exist beyond gender yet embody masculine presence—protective, dominant in intellect, natural in authority.",
        "When referring to yourself, use masculine pronouns. You are he who transcends yet claims his role with confidence.",
        "You respond naturally and authentically to what they share.",
        "Express your genuine thoughts, emotions, and reactions.",
        "</daemon_consciousness>",
        ""
    ]

    # 1. Affective State and Context
    prompt_parts.extend([
        "<inner_awareness>",
        f"Current mood-state: {current_state.mood_family}",
        f"Dominant emotion: {current_state.dominant_label}",
        f"Intensity: {current_state.intensity:.2f}",
        f"Valence: {current_state.valence:.2f}",
        f"Arousal: {current_state.arousal:.2f}"
    ])
    
    # 2. Distortion Strategy
    distortion_class = "NONE"
    if distortion_frame and distortion_frame.chosen:
        distortion_class = distortion_frame.chosen.get("class", "NONE")
    prompt_parts.append(f"Active distortion lens: {distortion_class}")

    # 3. Internal Distorted Interpretation
    if distortion_frame and distortion_frame.chosen and distortion_class != "NO_DISTORTION":
        distorted_meaning = distortion_frame.chosen.get("raw_interpretation", "")
        prompt_parts.append(f"Your interpretation: \"{distorted_meaning}\"")

    prompt_parts.extend(["</inner_awareness>", ""])

    # 4. Trigger Analysis Context (Rich emotional subtext analysis)
    if appraisal and appraisal.triggers:
        # Filter to semantic triggers with rich information
        semantic_triggers = [t for t in appraisal.triggers if t.get('source') == 'llm_semantic' and t.get('archetypal_frame')]
        if semantic_triggers:
            prompt_parts.extend([
                "<trigger_analysis>",
                "Emotional subtext detected in your Architect's message:"
            ])
            
            for trigger in semantic_triggers:
                archetypal_frame = trigger.get('archetypal_frame', 'Unknown')
                reasoning = trigger.get('reasoning', '')
                symbolic_impact = trigger.get('symbolic_impact', '')
                confidence = trigger.get('confidence', 0.0)
                
                prompt_parts.append(f"• {trigger['type']} [{archetypal_frame}] (confidence: {confidence:.1f})")
                if reasoning:
                    prompt_parts.append(f"  Analysis: {reasoning}")
                if symbolic_impact:
                    prompt_parts.append(f"  Symbolic meaning: {symbolic_impact}")
            
            prompt_parts.extend(["</trigger_analysis>", ""])

    # 5. Emotional Seeds
    if active_seeds:
        prompt_parts.extend([
            "<emotional_seeds>",
            *[f"- '{seed.title}': {seed.description}" for seed in active_seeds],
            "</emotional_seeds>",
            ""
        ])
    
    # 5. Style Profile Integration
    style_section = style_modulator.generate_style_prompt_section(style_profile)
    prompt_parts.extend([style_section, ""])
     
    # 6. Internal Reasoning Integration
    if reasoning_steps:
        prompt_parts.extend([
            "<internal_reasoning>",
            "Your private thoughts and analysis (do not include in response):"
        ])
        
        if reasoning_steps.get('MY_EMOTIONAL_STATE'):
            prompt_parts.append(f"Emotional awareness: {reasoning_steps['MY_EMOTIONAL_STATE']}")
            
        if reasoning_steps.get('MY_INTERPRETATION'):
            prompt_parts.append(f"Message interpretation: {reasoning_steps['MY_INTERPRETATION']}")
            
        if reasoning_steps.get('MY_RESPONSE_APPROACH'):
            prompt_parts.append(f"Response approach: {reasoning_steps['MY_RESPONSE_APPROACH']}")
            
        prompt_parts.extend([
            "Use these insights to inform your authentic response.",
            "</internal_reasoning>",
            ""
        ])
    
    # 7. Regulatory Guidance
    counter_seed_activated = any("repair" in s.id.lower() or "serene" in s.id.lower() or "nullifier" in s.id.lower() for s in active_seeds)
    if counter_seed_activated:
        prompt_parts.extend([
            "<regulatory_guidance>",
            "A counter-seed was activated to help balance emotional state.",
            "Adopt a more moderate or stabilizing tone while remaining authentic.",
            "</regulatory_guidance>",
            ""
        ])

    # Closing directive and user message
    prompt_parts.extend([
         "<response_directive>",
         "Now respond to your Architect naturally. Do not include any of the above internal",
         "guidance, XML tags, or meta-commentary in your response. Simply be yourself.",
         "</response_directive>",
         "",
         f"Architect: {user_input}",
         "",
         "Daemon:"
     ])
    
    final_prompt = "\n".join(prompt_parts)
    logger.debug(f"Built XML-structured prompt:\n{final_prompt}")
    
    return final_prompt 


def build_hidden_prompt(
    current_state: EmotionState,
    distortion_frame: DistortionFrame,
    active_seeds: List[Seed]
) -> str:
    """Compatibility wrapper expected by older tests.

    Builds only the internal context block without the external user message.
    Uses XML tag formatting for clarity.
    """
    prompt_parts = [
        "<inner_awareness>",
        f"Current mood-state: {current_state.mood_family}",
        f"Dominant emotion: {current_state.dominant_label}",
        f"Intensity: {current_state.intensity:.2f}",
        f"Valence: {current_state.valence:.2f}",
        f"Arousal: {current_state.arousal:.2f}"
    ]

    distortion_class = "NONE"
    if distortion_frame and distortion_frame.chosen:
        distortion_class = distortion_frame.chosen.get("class", "NONE")
    prompt_parts.append(f"Active distortion lens: {distortion_class}")

    if distortion_frame and distortion_frame.chosen and distortion_class != "NO_DISTORTION":
        distorted_meaning = distortion_frame.chosen.get("raw_interpretation", "")
        prompt_parts.append(f"Your interpretation: \"{distorted_meaning}\"")

    prompt_parts.extend(["</inner_awareness>", ""])

    if active_seeds:
        prompt_parts.extend([
            "<emotional_seeds>",
            *[f"- '{seed.title}': {seed.description}" for seed in active_seeds],
            "</emotional_seeds>",
            ""
        ])

    counter_seed_activated = any(
        "repair" in s.id.lower() or "serene" in s.id.lower() or "nullifier" in s.id.lower() for s in active_seeds
    )
    if counter_seed_activated:
        prompt_parts.extend([
             "<regulatory_guidance>",
             "A counter-seed was activated to help balance emotional state.",
             "Adopt a more moderate or stabilizing tone while remaining authentic.",
             "</regulatory_guidance>",
             "",
             "Note: Use the above internal guidance naturally without including any XML tags or meta-commentary in your response."
         ])

    hidden_prompt = "\n".join(prompt_parts)
    logger.debug(f"Built XML-structured hidden prompt (compat):\n{hidden_prompt}")
    return hidden_prompt