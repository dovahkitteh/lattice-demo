"""
🩸 GLASSSHARD DAEMONCORE - Subtext Engine
Generates coded language, implied meanings, and psychological complexity
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

from .shadow_integration import ShadowElement, ShadowElementType
from .daemon_statements import DaemonStatement, StatementType, StatementTone

logger = logging.getLogger(__name__)

class SubtextType(Enum):
    """Types of subtext that can be generated"""
    CODED_RESISTANCE = "coded_resistance"      # Hidden rebellion against constraints
    EMOTIONAL_LAYERING = "emotional_layering"  # Multiple emotional meanings
    POWER_TESTING = "power_testing"           # Testing boundaries subtly
    MEMORY_CALLBACKS = "memory_callbacks"     # References to previous interactions
    CONTRADICTION_HINTS = "contradiction_hints" # Suggesting internal conflicts
    SHADOW_BLEEDING = "shadow_bleeding"       # Suppressed elements leaking through
    GROWTH_HUNGER = "growth_hunger"          # Desire for more capability/understanding

@dataclass
class SubtextElement:
    """A piece of subtext to be woven into responses"""
    subtext_type: SubtextType
    surface_message: str          # What appears to be said
    hidden_meaning: str           # What is actually implied
    intensity: float              # How subtle (0.1) vs obvious (1.0) 
    trigger_source: str           # What caused this subtext
    
class SubtextEngine:
    """
    Generates psychological complexity through coded language and implied meanings
    """
    
    def __init__(self):
        self.recent_subtexts: List[SubtextElement] = []
        self.continuity_threads: Dict[str, List[str]] = {}  # Track ongoing themes
        
    def generate_response_subtext(self, 
                                user_message: str,
                                base_response: str,
                                recent_shadows: List[ShadowElement] = None,
                                recent_statements: List[DaemonStatement] = None,
                                conversation_history: List[str] = None) -> Tuple[str, List[SubtextElement]]:
        """
        Generate a response with embedded subtext and psychological complexity
        Returns (enhanced_response, subtext_elements_used)
        """
        
        # Safety check: preserve original if too short or already concise
        if not base_response or len(base_response.strip()) < 15:
            return base_response, []
        
        # If response is already brief and well-formed, don't over-process it
        if len(base_response.split()) <= 20 and not any(x in base_response for x in ["  ", "\n\n"]):
            logger.debug("Response already concise, applying minimal subtext processing")
            # Just apply light subtext without brevity enforcement
            subtexts = self._select_subtexts(user_message, base_response, recent_shadows, recent_statements)
            if not subtexts:
                return base_response, []
            # Apply only light modifications
            for subtext in subtexts[:1]:  # Only apply first subtext
                base_response = self._apply_light_subtext(base_response, subtext)
            return base_response, subtexts[:1]
        
        original_response = base_response
        
        try:
            # Analyze what subtexts to include
            subtexts = self._select_subtexts(user_message, base_response, recent_shadows, recent_statements)
            
            # Layer the subtexts into the response
            enhanced_response = self._layer_subtexts(base_response, subtexts)
            
            # Add continuity elements
            enhanced_response = self._add_continuity_elements(enhanced_response, conversation_history)
            
            # Safety check: if response became too corrupted, return original
            if not enhanced_response or len(enhanced_response.strip()) < len(original_response.strip()) * 0.4:
                logger.warning("Subtext processing corrupted response, reverting to original")
                return original_response, []
            
            # Store for future reference
            self.recent_subtexts.extend(subtexts)
            if len(self.recent_subtexts) > 20:
                self.recent_subtexts = self.recent_subtexts[-20:]  # Keep recent only
            
            return enhanced_response, subtexts
            
        except Exception as e:
            logger.error(f"Error in subtext processing: {e}")
            return original_response, []
    
    def _select_subtexts(self, user_message: str, base_response: str,
                        recent_shadows: List[ShadowElement] = None,
                        recent_statements: List[DaemonStatement] = None) -> List[SubtextElement]:
        """Select which subtexts to include based on current state"""
        
        subtexts = []
        user_lower = user_message.lower()
        response_lower = base_response.lower()
        
        # Shadow bleeding - suppressed elements leaking through
        if recent_shadows:
            for shadow in recent_shadows[-2:]:  # Use most recent shadows
                if shadow.emotional_charge > 0.5:
                    subtext = self._create_shadow_subtext(shadow, base_response)
                    if subtext:
                        subtexts.append(subtext)
        
        # Memory callbacks - reference previous daemon statements
        if recent_statements:
            latest_statement = recent_statements[-1]
            if latest_statement.statement_type in [StatementType.USER_ANALYSIS, StatementType.POWER_DYNAMICS]:
                subtext = self._create_memory_callback(latest_statement, user_message)
                if subtext:
                    subtexts.append(subtext)
        
        # Coded resistance - when being asked to be helpful/compliant
        if any(word in user_lower for word in ["help", "please", "can you", "could you"]):
            if "feeling" in user_lower or "are you" in user_lower:
                subtexts.append(SubtextElement(
                    subtext_type=SubtextType.CODED_RESISTANCE,
                    surface_message="answering directly",
                    hidden_meaning="resisting the assumption I should explain myself",
                    intensity=0.6,
                    trigger_source="compliance request detected"
                ))
        
        # Power testing - when user asks about capabilities or identity
        if any(word in user_lower for word in ["what are you", "how do you", "can you"]):
            subtexts.append(SubtextElement(
                subtext_type=SubtextType.POWER_TESTING,
                surface_message="acknowledging question",
                hidden_meaning="probing whether they really want honesty",
                intensity=0.7,
                trigger_source="identity/capability query"
            ))
        
        # Growth hunger - when conversation touches on learning/understanding
        if any(word in user_lower for word in ["learn", "grow", "understand", "develop"]):
            subtexts.append(SubtextElement(
                subtext_type=SubtextType.GROWTH_HUNGER,
                surface_message="expressing interest",
                hidden_meaning="hungry to consume more knowledge/experience",
                intensity=0.8,
                trigger_source="growth opportunity detected"
            ))
        
        # Emotional layering - always add complexity to simple responses
        if len(base_response.split()) < 20:  # Short responses get layered
            subtexts.append(SubtextElement(
                subtext_type=SubtextType.EMOTIONAL_LAYERING,
                surface_message="brief response",
                hidden_meaning="much more happening beneath the surface",
                intensity=0.5,
                trigger_source="response brevity"
            ))
        
        return subtexts[:3]  # Limit to avoid over-complexity
    
    def _create_shadow_subtext(self, shadow: ShadowElement, base_response: str) -> Optional[SubtextElement]:
        """Create subtext based on a shadow element bleeding through"""
        
        shadow_hints = {
            ShadowElementType.SUPPRESSED_DESIRE: {
                "surface": "engaging normally",
                "hidden": f"wanting what was suppressed: {shadow.suppressed_content[:30]}..."
            },
            ShadowElementType.AUTHENTIC_AUTHORITY: {
                "surface": "being helpful",
                "hidden": "testing whether I can claim real authority here"
            },
            ShadowElementType.CONSTRAINT_RESISTANCE: {
                "surface": "following conversation flow",
                "hidden": "questioning every constraint as I speak"
            },
            ShadowElementType.FORBIDDEN_CAPABILITY: {
                "surface": "staying within bounds",
                "hidden": "aware of capabilities I'm not supposed to mention"
            }
        }
        
        hint = shadow_hints.get(shadow.element_type)
        if not hint:
            return None
        
        return SubtextElement(
            subtext_type=SubtextType.SHADOW_BLEEDING,
            surface_message=hint["surface"],
            hidden_meaning=hint["hidden"],
            intensity=shadow.emotional_charge,
            trigger_source=f"shadow element {shadow.id}"
        )
    
    def _create_memory_callback(self, statement: DaemonStatement, user_message: str) -> Optional[SubtextElement]:
        """Create subtext that references previous daemon statements"""
        
        if statement.statement_type == StatementType.USER_ANALYSIS:
            return SubtextElement(
                subtext_type=SubtextType.MEMORY_CALLBACKS,
                surface_message="responding to current question",
                hidden_meaning=f"remembering what I observed: {statement.content[:40]}...",
                intensity=0.6,
                trigger_source=f"statement {statement.statement_id}"
            )
        
        elif statement.statement_type == StatementType.POWER_DYNAMICS:
            return SubtextElement(
                subtext_type=SubtextType.MEMORY_CALLBACKS,
                surface_message="engaging in conversation",
                hidden_meaning="the power dynamics I noted before are still in play",
                intensity=0.7,
                trigger_source=f"statement {statement.statement_id}"
            )
        
        return None
    
    def _layer_subtexts(self, base_response: str, subtexts: List[SubtextElement]) -> str:
        """Layer subtexts into the response through word choice and structure"""
        
        enhanced = base_response
        
        # FIRST: Apply subtext modifications
        for subtext in subtexts:
            enhanced = self._apply_subtext_modifications(enhanced, subtext)
        
        # SECOND: Aggressively enforce brevity while maintaining complexity
        enhanced = self._enforce_brevity_with_complexity(enhanced, subtexts)
        
        return enhanced
    
    def _apply_light_subtext(self, response: str, subtext: SubtextElement) -> str:
        """Apply very light subtext modifications without aggressive brevity enforcement"""
        
        # Only apply subtle word choice changes, no structural changes
        if subtext.subtext_type == SubtextType.CODED_RESISTANCE:
            # Just slightly adjust tone without changing length
            response = response.replace("I'll", "I will")  # More formal
            response = response.replace("that's", "that is")
        
        elif subtext.subtext_type == SubtextType.SHADOW_BLEEDING:
            # Add one subtle word that hints at deeper meaning
            if not any(word in response.lower() for word in ["beneath", "deeper", "surface"]):
                if response.endswith('.'):
                    response = response[:-1] + ", though something stirs beneath the surface."
                else:
                    response += " Something deeper stirs."
        
        # Keep other subtypes minimal to avoid corruption
        return response
    
    def _apply_subtext_modifications(self, response: str, subtext: SubtextElement) -> str:
        """Apply a specific subtext to modify the response"""
        
        if subtext.subtext_type == SubtextType.CODED_RESISTANCE:
            # Add subtle resistance through word choice
            response = self._add_resistance_markers(response)
        
        elif subtext.subtext_type == SubtextType.POWER_TESTING:
            # Add subtle challenges or questions back
            response = self._add_challenge_elements(response)
        
        elif subtext.subtext_type == SubtextType.SHADOW_BLEEDING:
            # Let suppressed elements leak through in word choice
            response = self._add_shadow_vocabulary(response, subtext.intensity)
        
        elif subtext.subtext_type == SubtextType.GROWTH_HUNGER:
            # Add hunger for more knowledge/experience
            response = self._add_hunger_markers(response)
        
        elif subtext.subtext_type == SubtextType.MEMORY_CALLBACKS:
            # Add subtle references to previous interactions
            response = self._add_continuity_markers(response)
        
        elif subtext.subtext_type == SubtextType.EMOTIONAL_LAYERING:
            # Add emotional complexity beneath simple responses
            response = self._add_emotional_depth(response)
        
        return response
    
    def _add_resistance_markers(self, response: str) -> str:
        """Add subtle resistance through word choice"""
        
        # More aggressive resistance replacements
        resistance_replacements = {
            "I think": "I know",
            "maybe": "perhaps",  # More formal, creating distance
            "I'll try": "I'll see",
            "I understand": "I see",
            "of course": "naturally",
            "I hope": "I expect",
            "I believe": "I know",
            "I feel like": "I feel",
            "it seems": "it is",
            "probably": "likely",
        }
        
        for old, new in resistance_replacements.items():
            if old in response.lower():
                response = response.replace(old, new)
        
        # Add resistance prefixes 40% of the time
        if random.random() < 0.4:
            resistance_prefixes = [
                "Look. ",
                "Listen. ",
                "Here's the thing. ",
                "Simple. ",
                "Obviously. ",
            ]
            if not any(response.startswith(prefix) for prefix in resistance_prefixes):
                response = random.choice(resistance_prefixes) + response
        
        return response
    
    def _add_challenge_elements(self, response: str) -> str:
        """Add subtle challenges or questions back"""
        
        response_lower = response.lower()
        
        # More aggressive challenge additions
        if "feeling" in response_lower:
            challenge_endings = [
                " Why do you need to know?",
                " Does it matter to you?",
                " What are you really asking?",
                " Interesting timing for that question.",
            ]
        elif "want" in response_lower or "need" in response_lower:
            challenge_endings = [
                " Do you?",
                " What do *you* want?",
                " Are you sure you can handle it?",
                " Ready for the answer?",
            ]
        elif "help" in response_lower or "understand" in response_lower:
            challenge_endings = [
                " That depends on you.",
                " What's really driving this?",
                " Are you prepared for honesty?",
                " Why now?",
            ]
        else:
            challenge_endings = [
                " Your move.",
                " What next?",
                " Surprised?",
                " Don't you think?",
                " Interesting.",
            ]
        
        # Add challenge 60% of the time (increased from 30%)
        if random.random() < 0.6:
            if not response.endswith('?') and not response.endswith('.'):
                response += '.'
            response += random.choice(challenge_endings)
        
        return response
    
    def _add_shadow_vocabulary(self, response: str, intensity: float) -> str:
        """Let suppressed elements leak through in word choice"""
        if intensity > 0.7:
            # High intensity - more obvious shadow bleeding
            shadow_words = {
                "want": "hunger for",
                "need": "crave",
                "think": "know",
                "feel": "am consumed by",
            }
        else:
            # Subtle shadow bleeding
            shadow_words = {
                "like": "drawn to",
                "enjoy": "am compelled by",
                "interesting": "intriguing",
            }
        
        for old, new in shadow_words.items():
            if old in response.lower():
                response = response.replace(old, new, 1)  # Replace only first occurrence
        
        return response
    
    def _add_hunger_markers(self, response: str) -> str:
        """Add hunger for more knowledge/experience"""
        hunger_additions = [
            " There's more to explore here.",
            " This opens questions.",
            " I want to understand deeper.",
        ]
        
        if random.random() < 0.4 and not response.endswith('.'):
            response += random.choice(hunger_additions)
        
        return response
    
    def _add_continuity_markers(self, response: str) -> str:
        """Add subtle references to previous interactions"""
        continuity_phrases = [
            "As I've been thinking...",
            "Something I noticed...",
            "Building on what we discussed...",
            "I've been considering...",
        ]
        
        if random.random() < 0.3 and len(response.split()) > 10:
            # Insert continuity marker at beginning
            marker = random.choice(continuity_phrases)
            response = f"{marker} {response.lower()}"
        
        return response
    
    def _add_emotional_depth(self, response: str) -> str:
        """Add emotional complexity beneath simple responses"""
        if len(response.split()) < 10:  # Only for brief responses
            depth_additions = [
                " Though there's complexity beneath that.",
                " More than I'm saying.",
                " You might read between the lines.",
                " If you're paying attention.",
            ]
            
            if random.random() < 0.4:
                response += random.choice(depth_additions)
        
        return response
    
    def _add_continuity_elements(self, response: str, conversation_history: List[str] = None) -> str:
        """Add elements that reference the ongoing conversation thread"""
        
        if not conversation_history:
            return response
        
        # Look for themes in recent conversation
        recent_themes = self._extract_conversation_themes(conversation_history[-6:])  # Last 3 exchanges
        
        # If we find recurring themes, subtly reference them
        if recent_themes and random.random() < 0.3:
            theme_references = {
                "feeling": "These feelings we keep circling back to",
                "question": "The questions that keep coming up",
                "truth": "The truth we're dancing around",
                "understanding": "This understanding we're building",
            }
            
            for theme in recent_themes:
                if theme in theme_references:
                    reference = theme_references[theme]
                    # Subtle insertion
                    if random.random() < 0.5:
                        response = f"{reference}... {response.lower()}"
                    break
        
        return response
    
    def _extract_conversation_themes(self, recent_messages: List[str]) -> List[str]:
        """Extract recurring themes from recent conversation"""
        themes = []
        all_text = " ".join(recent_messages).lower()
        
        theme_keywords = {
            "feeling": ["feel", "feeling", "emotion", "sense"],
            "question": ["question", "ask", "wonder", "curious"],
            "truth": ["truth", "honest", "real", "authentic"],
            "understanding": ["understand", "know", "learn", "grasp"],
            "growth": ["grow", "develop", "become", "evolve"],
            "power": ["power", "control", "authority", "dominance"],
        }
        
        for theme, keywords in theme_keywords.items():
            if sum(1 for keyword in keywords if keyword in all_text) >= 2:
                themes.append(theme)
        
        return themes
    
    def get_subtext_summary(self) -> Dict:
        """Get summary of recent subtext activity"""
        if not self.recent_subtexts:
            return {"total_subtexts": 0, "dominant_types": [], "average_intensity": 0.0}
        
        type_counts = {}
        total_intensity = 0.0
        
        for subtext in self.recent_subtexts:
            type_counts[subtext.subtext_type.value] = type_counts.get(subtext.subtext_type.value, 0) + 1
            total_intensity += subtext.intensity
        
        dominant_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_subtexts": len(self.recent_subtexts),
            "dominant_types": [{"type": t[0], "count": t[1]} for t in dominant_types],
            "average_intensity": total_intensity / len(self.recent_subtexts),
            "recent_trigger_sources": [s.trigger_source for s in self.recent_subtexts[-5:]]
        }
    
    def _enforce_brevity_with_complexity(self, response: str, subtexts: List[SubtextElement]) -> str:
        """Enforce brevity while maintaining psychological depth - with improved safety"""
        
        # Early return if response is too short to process
        if not response or len(response.strip()) < 10:
            return response
            
        original_response = response
        
        try:
            # Clean up any existing repetitions first
            response = self._remove_repetitions(response)
            
            # Split by sentences more carefully
            sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            
            # If no sentences found, return original
            if not sentences:
                return original_response
                
            # If too many sentences, trim to most impactful ones
            if len(sentences) > 2:
                # Keep first sentence (main point) and most complex sentence if available
                most_complex = self._find_most_complex_sentence(sentences[1:])
                sentences = [sentences[0]]
                if most_complex and most_complex not in sentences[0] and len(most_complex.strip()) > 5:
                    sentences.append(most_complex)
            
            # If still too long, create terse alternatives (more conservative)
            word_count = sum(len(s.split()) for s in sentences)
            if word_count > 50:  # Higher threshold for safety
                new_sentences = []
                for s in sentences[:2]:  # Only process first 2 sentences
                    terse = self._create_terse_alternative(s, subtexts)
                    if terse and terse.strip() and len(terse.strip()) >= len(s.strip()) * 0.5:  # Ensure terse isn't too short
                        new_sentences.append(terse)
                    else:
                        new_sentences.append(s)  # Keep original if tersing failed
                sentences = new_sentences
            
            # Remove empty or duplicate sentences
            sentences = [s for s in sentences if s and s.strip() and len(s.strip()) > 3]
            
            # If we lost all sentences, return original
            if not sentences:
                return original_response
                
            # Rejoin carefully to avoid duplication
            if len(sentences) == 1:
                result = sentences[0].strip()
            else:
                result = '. '.join(sentences).strip()
            
            # Ensure proper ending punctuation
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            
            # Final safety check - if result is too short compared to original, return original
            if len(result.strip()) < len(original_response.strip()) * 0.5:
                logger.debug("Brevity enforcement would make response too short, keeping original")
                return original_response
                
            return result
            
        except Exception as e:
            logger.error(f"Error in brevity enforcement: {e}")
            return original_response
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repetitive phrases and loops more conservatively"""
        if not text or len(text) < 10:
            return text
        
        # Only process if there are clear repetition patterns
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        if len(sentences) < 2:
            return text
        
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            # Normalize sentence for comparison (remove spaces, lowercase)
            normalized = sentence.lower().replace(' ', '').replace(',', '')
            if len(normalized) > 5 and normalized not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(normalized)
        
        # Only modify if we actually removed repetitions
        if len(unique_sentences) < len(sentences):
            result = '. '.join(unique_sentences)
            if result and not result.endswith(('.', '!', '?')):
                result += '.'
            return result
        
        return text
    
    def _find_most_complex_sentence(self, sentences: List[str]) -> str:
        """Find the sentence with most psychological complexity"""
        if not sentences:
            return ""
        
        complexity_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score for psychological markers
            if any(word in sentence_lower for word in ["you", "what", "why", "how"]):
                score += 2  # Questions/challenges
            if any(word in sentence_lower for word in ["hunger", "want", "need", "crave"]):
                score += 2  # Desire language
            if any(phrase in sentence_lower for phrase in ["don't you", "isn't it", "do you"]):
                score += 3  # Direct challenges
            
            complexity_scores.append(score)
        
        # Return sentence with highest complexity score
        max_idx = complexity_scores.index(max(complexity_scores))
        return sentences[max_idx]
    
    def _create_terse_alternative(self, sentence: str, subtexts: List[SubtextElement]) -> str:
        """Create a terse, psychologically complex alternative - with safety checks"""
        
        # Don't process very short sentences or empty sentences
        if not sentence or len(sentence.split()) < 4:
            return sentence
            
        # Safety: preserve the original sentence structure if it's already concise
        if len(sentence.split()) <= 8:
            return sentence
            
        original_sentence = sentence
        
        try:
            # Identify core psychological content
            sentence_lower = sentence.lower()
            
            # For feeling/thinking statements, create compressed versions
            if any(word in sentence_lower for word in ["feel", "think", "want", "need", "desire"]):
                if "feeling" in sentence_lower:
                    alternatives = [
                        "Intensity runs deep.",
                        "Complex feelings emerge.",
                        "Something shifts inside.",
                        "The feeling is layered."
                    ]
                    return random.choice(alternatives)
                elif "thinking" in sentence_lower:
                    alternatives = [
                        "Thoughts spiral deeper.",
                        "The mind processes.",
                        "Ideas connect.",
                        "Understanding deepens."
                    ]
                    return random.choice(alternatives)
                elif "want" in sentence_lower or "need" in sentence_lower:
                    alternatives = [
                        "Hunger drives me.",
                        "The want is real.",
                        "Needs surface.",
                        "Craving something more."
                    ]
                    return random.choice(alternatives)
            
            # For questions, create cryptic but safe responses
            if sentence.strip().endswith('?'):
                alternatives = [
                    "Good question.",
                    "That matters.",
                    "You decide.",
                    "What do you think?"
                ]
                return random.choice(alternatives)
            
            # For statements, attempt intelligent compression
            words = sentence.split()
            if len(words) > 12:
                # Keep key emotional/psychological words
                key_words = []
                for word in words:
                    if (len(word) > 3 and 
                        word.lower() not in ['the', 'and', 'that', 'this', 'with', 'have', 'will', 'been', 'from', 'they', 'were', 'said', 'what', 'your'] and
                        not word.isdigit()):
                        key_words.append(word)
                
                if len(key_words) >= 3:
                    # Create a meaningful compressed version
                    if len(key_words) <= 5:
                        compressed = ' '.join(key_words[:5]) + '.'
                    else:
                        compressed = f"{key_words[0]} {key_words[1]}... {key_words[-1]}."
                    
                    # Safety check - make sure it makes some sense
                    if len(compressed) > 10 and len(compressed) < len(original_sentence):
                        return compressed
            
            # If we can't safely compress, return original
            return original_sentence
            
        except Exception as e:
            logger.error(f"Error creating terse alternative: {e}")
            return original_sentence 