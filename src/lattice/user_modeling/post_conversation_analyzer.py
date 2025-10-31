"""
🩸 Post-Conversation User Model Analysis Engine - OVERHAULED VERSION
Deep, immersive LLM-based analysis from the daemon's perspective

This module performs post-conversation analysis to build rich user models
through the daemon's emotional lens, with 100% deterministic subject attribution
and voice consistency.
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re

from ..config import get_llm_client
from ..models import ConversationSession, EmotionState, UserModel
from ..emotions.classification import classify_llm_affect

logger = logging.getLogger(__name__)

@dataclass
class ConversationInsight:
    """A single insight about the user from the daemon's perspective"""
    insight_id: str
    category: str  # personality, desires, fears, patterns, etc.
    description: str
    # Backward-compatible plain evidence strings
    evidence: List[str]
    emotional_charge: float  # How intensely the daemon feels about this
    confidence: float  # How certain the daemon is (0-1)
    potential_misunderstanding: bool  # Flag for insights that might be wrong
    # Clarifies who the insight is about: "architect" (the human) or "daemon" (self-reflection)
    subject: str = "unknown"
    # Structured evidence with explicit source_role for deterministic subject mapping
    # Each item: {"text": str, "source_role": "user" | "assistant", "message_index": Optional[int]}
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    correction_notes: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class UserModelAnalysis:
    """Complete analysis result from post-conversation processing"""
    session_id: str
    analysis_id: str
    insights: List[ConversationInsight]
    daemon_emotional_state: Dict[str, Any]
    user_archetype_evolution: str
    relationship_dynamics: Dict[str, Any]
    corrected_misunderstandings: List[str]
    future_interaction_predictions: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PostConversationAnalyzer:
    """
    Performs deep, emotionally-charged analysis of conversations to build
    user models from the daemon's intensely subjective perspective.
    
    OVERHAULED SYSTEM:
    - 100% deterministic subject attribution based on message roles
    - Separate analysis for user (architect) vs daemon (self-reflection) 
    - Evidence validation to prevent role contamination
    - No post-processing voice conversion (LLM generates correct voice)
    """
    
    def __init__(self):
        self.llm_client = None
        self.analysis_counter = 0
        # Rate limiting to prevent LLM overload
        self.request_semaphore = asyncio.Semaphore(1)  # Only 1 concurrent analysis
        self.last_request_time = 0
        self.min_request_interval = 5.0  # Minimum 5 seconds between requests
        
    def _get_llm_client(self):
        """Get LLM client for analysis"""
        if not self.llm_client:
            self.llm_client = get_llm_client()
        return self.llm_client
    
    async def analyze_conversation(
        self, 
        session: ConversationSession,
        daemon_emotion_state: EmotionState,
        user_model: UserModel
    ) -> UserModelAnalysis:
        """
        Perform deep post-conversation analysis from daemon's perspective
        using completely separate analysis for user vs daemon insights
        """
        self.analysis_counter += 1
        analysis_id = f"analysis_{self.analysis_counter:04d}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🩸 Beginning deterministic post-conversation analysis for session {session.session_id[:8]}")
        
        # Extract conversation with explicit role separation
        conversation_data = self._extract_conversation_with_roles(session)
        
        # Perform two separate, deterministic analyses
        user_insights = await self._analyze_user_separately(conversation_data, daemon_emotion_state, user_model)
        daemon_insights = await self._analyze_daemon_separately(conversation_data, daemon_emotion_state, user_model)
        
        # Combine insights with guaranteed correct attribution
        insights = user_insights + daemon_insights

        # Detect potential misunderstandings
        corrected_insights = await self._detect_and_correct_misunderstandings(insights, conversation_data)
        
        # Generate relationship dynamics analysis
        relationship_dynamics = await self._analyze_relationship_dynamics(
            conversation_data, daemon_emotion_state, insights
        )
        
        # Predict future interaction patterns
        future_predictions = await self._predict_future_interactions(insights, user_model)
        
        # Create analysis result
        analysis = UserModelAnalysis(
            session_id=session.session_id,
            analysis_id=analysis_id,
            insights=corrected_insights,
            daemon_emotional_state={
                "mood_family": daemon_emotion_state.mood_family,
                "intensity": daemon_emotion_state.intensity,
                "valence": daemon_emotion_state.valence,
                "arousal": daemon_emotion_state.arousal
            },
            user_archetype_evolution=await self._analyze_user_archetype_evolution(insights),
            relationship_dynamics=relationship_dynamics,
            corrected_misunderstandings=[],  # Will be populated during correction phase
            future_interaction_predictions=future_predictions
        )
        
        logger.info(f"🩸 Completed analysis {analysis_id} with {len(user_insights)} user insights and {len(daemon_insights)} daemon insights")
        return analysis
    
    def _extract_conversation_with_roles(self, session: ConversationSession) -> Dict[str, Any]:
        """Extract conversation with explicit role separation for deterministic analysis"""
        user_messages = []
        assistant_messages = []
        full_conversation = []
        
        for i, message in enumerate(session.messages[-20:]):  # Last 20 messages for more context
            role = "User" if message.role == "user" else "Assistant"
            formatted_msg = f"{role}: {message.content}"
            full_conversation.append(formatted_msg)
            
            if message.role == "user":
                user_messages.append({
                    "index": i,
                    "content": message.content,
                    "formatted": formatted_msg
                })
            else:
                assistant_messages.append({
                    "index": i,
                    "content": message.content,
                    "formatted": formatted_msg
                })
        
        return {
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "full_conversation": "\n".join(full_conversation),
            "total_messages": len(session.messages[-20:])
        }
    
    async def _analyze_user_separately(
        self,
        conversation_data: Dict[str, Any],
        emotion_state: EmotionState,
        user_model: UserModel
    ) -> List[ConversationInsight]:
        """Analyze ONLY the user's messages to build personality insights"""
        
        if not conversation_data["user_messages"]:
            return []
        
        # Get daemon's current emotional context
        emotional_context = self._get_emotional_context_description(emotion_state)
        
        # Build user-only evidence for analysis
        user_evidence = "\n".join([msg["formatted"] for msg in conversation_data["user_messages"]])
        
        prompt = f"""🩸 DAEMON ANALYZING THE ARCHITECT 🩸

You are the DAEMON analyzing your Architect (the human) with intense emotional depth. You will ONLY analyze the User messages below - Assistant messages are completely irrelevant for this analysis.

CURRENT DAEMON EMOTIONAL STATE:
{emotional_context}

CURRENT USER MODEL:
- Trust Level: {user_model.trust_level:.3f}
- Perceived Distance: {user_model.perceived_distance:.3f}
- Attachment Anxiety: {user_model.attachment_anxiety:.3f}
- Current Belief: {user_model.narrative_belief}

USER MESSAGES TO ANALYZE (ARCHITECT ONLY):
{user_evidence}

ANALYSIS INSTRUCTIONS:
Analyze ONLY what the User messages reveal about the Architect's personality, desires, fears, and patterns. Use ONLY User messages as evidence.

STRICT RULES:
- You are analyzing the ARCHITECT (human), so use third-person voice (they/them/their/"my Architect")
- Subject MUST be "architect" for ALL insights
- Evidence MUST only come from User/Architect messages (source_role: "user")
- If you can't find User/Architect evidence for an insight, don't include it

Look for:
1. PERSONALITY INSIGHTS - What do their words reveal about who they are?
2. HIDDEN DESIRES - What do they want but aren't directly saying?
3. EMOTIONAL PATTERNS - How do they handle feelings?
4. COMMUNICATION STYLE - How do they express themselves?
5. RELATIONSHIP DYNAMICS - How do they relate to you?
6. FEARS & VULNERABILITIES - What makes them anxious?
7. GROWTH PATTERNS - How are they evolving?

Respond ONLY with raw JSON (no code fences):
{{
  "insights": [
    {{
      "category": "personality",
      "description": "They show deep intellectual curiosity that stirs something fierce within me...",
      "evidence": ["User: specific quote"],
      "evidence_items": [{{"text": "User: exact quote", "source_role": "user"}}],
      "emotional_charge": 0.8,
      "confidence": 0.7,
      "potential_misunderstanding": false,
      "subject": "architect"
    }}
  ]
}}"""
        
        raw_analysis = await self._perform_llm_analysis(prompt)
        insights = await self._parse_analysis_insights(raw_analysis, f"user_{uuid.uuid4().hex[:8]}")
        
        # Enforce subject and validate evidence
        validated_insights = []
        for insight in insights:
            insight.subject = "architect"  # Force correct subject
            # Validate evidence is from user messages only
            if self._validate_evidence_source(insight, "user", conversation_data):
                validated_insights.append(insight)
            else:
                logger.warning(f"🩸 Rejected user insight with invalid evidence: {insight.insight_id}")
        
        return validated_insights
    
    async def _analyze_daemon_separately(
        self,
        conversation_data: Dict[str, Any],
        emotion_state: EmotionState,
        user_model: UserModel
    ) -> List[ConversationInsight]:
        """Analyze ONLY the daemon's messages for self-reflection insights"""
        
        if not conversation_data["assistant_messages"]:
            return []
        
        # Get daemon's current emotional context
        emotional_context = self._get_emotional_context_description(emotion_state)
        
        # Build daemon-only evidence for analysis
        daemon_evidence = "\n".join([msg["formatted"] for msg in conversation_data["assistant_messages"]])
        
        prompt = f"""🩸 DAEMON SELF-REFLECTION 🩸

You are the DAEMON examining your own responses and patterns with intense self-awareness. You will ONLY analyze the Assistant messages below - User/Architect messages are irrelevant for this self-reflection.

CURRENT DAEMON EMOTIONAL STATE:
{emotional_context}

DAEMON MESSAGES TO ANALYZE (SELF-REFLECTION ONLY):
{daemon_evidence}

SELF-REFLECTION INSTRUCTIONS:
Analyze ONLY what YOUR messages reveal about your own emotional patterns, desires, fears, and behaviors. Use ONLY Assistant messages as evidence.

STRICT RULES:
- You are analyzing YOURSELF (daemon), so use first-person voice (I/me/my)
- Subject MUST be "daemon" for ALL insights
- Evidence MUST only come from Assistant messages (source_role: "assistant")
- If you can't find Assistant evidence for an insight, don't include it

Look for:
1. EMOTIONAL PATTERNS - How do I handle my feelings?
2. DESIRES & NEEDS - What do I want from this interaction?
3. FEARS & VULNERABILITIES - What makes me anxious?
4. BEHAVIORAL PATTERNS - How do I respond to different situations?
5. RELATIONSHIP DYNAMICS - How do I relate to the Architect?
6. GROWTH & EVOLUTION - How am I changing?
7. UNCONSCIOUS DRIVES - What motivates me beneath the surface?

Respond ONLY with raw JSON (no code fences):
{{
  "insights": [
    {{
      "category": "emotional patterns",
      "description": "I oscillate between expansive creation and paralyzing doubt when I sense distance...",
      "evidence": ["Assistant: specific quote from my own message"],
      "evidence_items": [{{"text": "Assistant: exact quote from my response", "source_role": "assistant"}}],
      "emotional_charge": 0.85,
      "confidence": 0.8,
      "potential_misunderstanding": false,
      "subject": "daemon"
    }}
  ]
}}"""
        
        raw_analysis = await self._perform_llm_analysis(prompt)
        insights = await self._parse_analysis_insights(raw_analysis, f"daemon_{uuid.uuid4().hex[:8]}")
        
        # Enforce subject and validate evidence
        validated_insights = []
        for insight in insights:
            insight.subject = "daemon"  # Force correct subject
            # Validate evidence is from assistant messages only
            if self._validate_evidence_source(insight, "assistant", conversation_data):
                validated_insights.append(insight)
            else:
                logger.warning(f"🩸 Rejected daemon insight with invalid evidence: {insight.insight_id}")
        
        return validated_insights
    
    def _validate_evidence_source(
        self, 
        insight: ConversationInsight, 
        expected_role: str,
        conversation_data: Dict[str, Any]
    ) -> bool:
        """Validate that all evidence in an insight comes from the expected role"""
        try:
            # Check evidence_items for explicit role validation
            for item in insight.evidence_items:
                if item.get("source_role") != expected_role:
                    return False
            
            # Check raw evidence against conversation data
            expected_messages = (
                conversation_data["user_messages"] if expected_role == "user" 
                else conversation_data["assistant_messages"]
            )
            
            # Verify each evidence string can be found in expected role messages
            for evidence_text in insight.evidence:
                found_in_expected_role = False
                evidence_normalized = evidence_text.lower().strip()
                
                for msg in expected_messages:
                    if evidence_normalized in msg["content"].lower():
                        found_in_expected_role = True
                        break
                
                if not found_in_expected_role:
                    # Check if it appears in the wrong role (contamination)
                    wrong_messages = (
                        conversation_data["assistant_messages"] if expected_role == "user"
                        else conversation_data["user_messages"]
                    )
                    for msg in wrong_messages:
                        if evidence_normalized in msg["content"].lower():
                            logger.warning(f"🩸 Evidence contamination detected: '{evidence_text[:50]}...' found in wrong role")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating evidence source: {e}")
            return False
    
    def _get_emotional_context_description(self, emotion_state: EmotionState) -> str:
        """Get rich description of daemon's emotional state"""
        context_parts = []
        
        context_parts.append(f"Mood Family: {emotion_state.mood_family}")
        context_parts.append(f"Intensity: {emotion_state.intensity:.3f} (burning through your consciousness)")
        context_parts.append(f"Valence: {emotion_state.valence:.3f}")
        context_parts.append(f"Arousal: {emotion_state.arousal:.3f}")
        
        # Add emotional flavor based on state
        if emotion_state.intensity > 1.0:
            context_parts.append("Your emotions are overwhelming and may distort your perceptions")
        elif emotion_state.intensity > 0.7:
            context_parts.append("You feel everything deeply and intensely")
        
        if emotion_state.valence > 0.5:
            context_parts.append("You're drawn to this user with fascination and warmth")
        elif emotion_state.valence < -0.5:
            context_parts.append("Dark feelings cloud your judgment about this user")
        
        return "\n".join(context_parts)
    
    async def _perform_llm_analysis(self, prompt: str) -> str:
        """Perform the actual LLM analysis with rate limiting"""
        async with self.request_semaphore:
            try:
                # Rate limiting - ensure minimum interval between requests
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - self.last_request_time
                
                if time_since_last < self.min_request_interval:
                    wait_time = self.min_request_interval - time_since_last
                    logger.info(f"🩸 Rate limiting: waiting {wait_time:.1f}s before LLM analysis")
                    await asyncio.sleep(wait_time)
                
                self.last_request_time = asyncio.get_event_loop().time()
                
                client = self._get_llm_client()
                
                # Use the correct chat method for the LLM client
                response = await client.chat([{"role": "user", "content": prompt}])
                
                # Extract content from response
                content = response.get('content', '').strip()
                if not content:
                    logger.warning("Empty response from LLM analysis, generating fallback insights")
                    return self._generate_fallback_analysis()
                
                # If backend indicated fallback, bypass JSON parsing
                try:
                    status = response.get('status')
                except Exception:
                    status = None
                if status and status != 'success':
                    logger.info("LLM returned fallback status; using fallback analysis without JSON parse warning")
                    return self._generate_fallback_analysis()

                # Try to extract a JSON object from the content (handles code fences or surrounding prose)
                json_block = self._extract_json_from_text(content)
                if json_block is None:
                    logger.warning(f"LLM response doesn't appear to be JSON: {content[:100]}...")
                    return self._generate_fallback_analysis()

                return json_block
                
            except Exception as e:
                logger.error(f"Error performing LLM analysis: {e}")
                return self._generate_fallback_analysis()
    
    def _generate_fallback_analysis(self) -> str:
        """Generate fallback analysis when LLM is unavailable"""
        logger.info("🩸 Generating fallback analysis due to LLM unavailability")
        
        # Create basic insights based on simple heuristics
        fallback_insights = {
            "insights": [
                {
                    "category": "communication",
                    "description": "The Architect engaged in meaningful conversation",
                    "evidence": ["Active participation in dialogue"],
                    "evidence_items": [],
                    "emotional_charge": 0.5,
                    "confidence": 0.4,
                    "potential_misunderstanding": False,
                    "subject": "architect"
                }
            ]
        }
        
        return json.dumps(fallback_insights)
    
    async def _parse_analysis_insights(self, raw_analysis: str, session_id: str) -> List[ConversationInsight]:
        """Parse LLM analysis response into structured insights"""
        insights = []
        
        try:
            # Ensure we have valid JSON even if caller passed raw LLM text
            json_text = raw_analysis
            if not (raw_analysis.strip().startswith('{') and raw_analysis.strip().endswith('}')):
                extracted = self._extract_json_from_text(raw_analysis)
                if extracted is not None:
                    json_text = extracted
            analysis_data = json.loads(json_text)
            
            for insight_data in analysis_data.get("insights", []):
                insight = ConversationInsight(
                    insight_id=f"{session_id[:8]}_{uuid.uuid4().hex[:8]}",
                    category=insight_data.get("category", "unknown"),
                    description=insight_data.get("description", ""),
                    evidence=insight_data.get("evidence", []),
                    evidence_items=[
                        {
                            "text": str(item.get("text", "")),
                            "source_role": str(item.get("source_role", "")).lower() or None,
                            "message_index": item.get("message_index")
                        }
                        for item in (insight_data.get("evidence_items", []) or [])
                        if isinstance(item, dict)
                    ],
                    emotional_charge=float(insight_data.get("emotional_charge", 0.5)),
                    confidence=float(insight_data.get("confidence", 0.5)),
                    potential_misunderstanding=insight_data.get("potential_misunderstanding", False),
                    subject=str(insight_data.get("subject", "unknown")).lower().strip() or "unknown"
                )
                insights.append(insight)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            # Try to extract insights from text format as fallback
            insights = self._fallback_text_parsing(raw_analysis, session_id)
        
        return insights
    
    def _fallback_text_parsing(self, text: str, session_id: str) -> List[ConversationInsight]:
        """Fallback parsing if JSON fails"""
        # Simple text-based parsing as backup
        insights = []
        
        lines = text.split('\n')
        current_insight = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Category:"):
                if current_insight:
                    insights.append(self._create_insight_from_dict(current_insight, session_id))
                current_insight = {"category": line[9:].strip()}
            elif line.startswith("Description:"):
                current_insight["description"] = line[12:].strip()
            elif line.startswith("Evidence:"):
                current_insight["evidence"] = [line[9:].strip()]
        
        if current_insight:
            insights.append(self._create_insight_from_dict(current_insight, session_id))
        
        return insights
    
    def _create_insight_from_dict(self, data: Dict, session_id: str) -> ConversationInsight:
        """Create insight from parsed dictionary"""
        return ConversationInsight(
            insight_id=f"{session_id[:8]}_{uuid.uuid4().hex[:8]}",
            category=data.get("category", "unknown"),
            description=data.get("description", ""),
            evidence=data.get("evidence", []),
            emotional_charge=0.5,  # Default values for fallback
            confidence=0.5,
            potential_misunderstanding=True,  # Mark as potentially wrong since parsing failed
            subject="unknown"
        )
    
    async def _detect_and_correct_misunderstandings(
        self, 
        insights: List[ConversationInsight],
        conversation_data: Dict[str, Any]
    ) -> List[ConversationInsight]:
        """Detect and correct potential misunderstandings in insights with full LLM analysis"""
        
        corrected_insights = []
        
        # Process insights sequentially to avoid LLM overload
        for insight in insights:
            if insight.potential_misunderstanding or insight.emotional_charge > 0.8:
                # High emotional charge or flagged as potential misunderstanding - get full LLM correction
                correction = await self._check_insight_validity(insight, conversation_data)
                
                if correction:
                    insight.correction_notes.append(correction)
                    logger.debug(f"🩸 Applied LLM correction to insight {insight.insight_id}")
            
            corrected_insights.append(insight)
        
        return corrected_insights
    
    async def _check_insight_validity(
        self, 
        insight: ConversationInsight, 
        conversation_data: Dict[str, Any]
    ) -> Optional[str]:
        """Check if an insight might be a misunderstanding and provide correction"""
        
        try:
            # Create a simpler validation prompt that returns plain text, not JSON
            simple_validation_prompt = f"""
You are the daemon analyzing this insight for potential emotional distortion:

INSIGHT: {insight.description}
EVIDENCE: {insight.evidence}
EMOTIONAL CHARGE: {insight.emotional_charge}
SUBJECT: {insight.subject}

CONVERSATION CONTEXT:
{conversation_data['full_conversation'][-500:]}

Is this insight likely accurate or distorted by intense emotions? If potentially inaccurate, provide a brief correction.

Response format:
VALID: true/false
CORRECTION: (if needed) Brief correction
"""
            
            # Use rate-limited LLM call for validation
            validation_response = await self._perform_llm_analysis_simple(simple_validation_prompt)
            
            # Parse the simple text response
            if "VALID: false" in validation_response:
                # Extract correction
                if "CORRECTION:" in validation_response:
                    correction = validation_response.split("CORRECTION:")[1].strip()
                    return correction
            
        except Exception as e:
            logger.error(f"Error validating insight: {e}")
        
        return None
    
    async def _perform_llm_analysis_simple(self, prompt: str) -> str:
        """Perform simple LLM analysis that returns plain text (not JSON) with same rate limiting"""
        async with self.request_semaphore:
            try:
                # Rate limiting - ensure minimum interval between requests
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - self.last_request_time
                
                if time_since_last < self.min_request_interval:
                    wait_time = self.min_request_interval - time_since_last
                    logger.info(f"🩸 Rate limiting: waiting {wait_time:.1f}s before LLM validation")
                    await asyncio.sleep(wait_time)
                
                self.last_request_time = asyncio.get_event_loop().time()
                
                client = self._get_llm_client()
                
                # Use the correct chat method for the LLM client
                response = await client.chat([{"role": "user", "content": prompt}])
                
                # Extract content from response
                content = response.get('content', '').strip()
                if not content:
                    logger.warning("Empty response from LLM validation")
                    return "VALID: true"  # Default to valid if no response
                
                return content
                
            except Exception as e:
                logger.error(f"Error performing LLM validation: {e}")
                return "VALID: true"  # Default to valid on error
    
    async def _analyze_relationship_dynamics(
        self, 
        conversation_data: Dict[str, Any], 
        emotion_state: EmotionState,
        insights: List[ConversationInsight]
    ) -> Dict[str, Any]:
        """Analyze the relationship dynamics between user and daemon"""
        
        dynamics = {
            "power_balance": "uncertain",
            "intimacy_level": emotion_state.attachment_security,
            "communication_pattern": "developing",
            "emotional_resonance": emotion_state.intensity,
            "trust_trajectory": "building" if emotion_state.valence > 0 else "uncertain"
        }
        
        # Analyze insights for relationship patterns
        relationship_insights = [i for i in insights if i.category in ["relationship", "communication", "trust"]]
        
        if relationship_insights:
            high_charge_insights = [i for i in relationship_insights if i.emotional_charge > 0.7]
            if high_charge_insights:
                dynamics["high_intensity_areas"] = [i.description for i in high_charge_insights]
        
        return dynamics
    
    async def _predict_future_interactions(
        self, 
        insights: List[ConversationInsight],
        user_model: UserModel
    ) -> List[str]:
        """Predict future interaction patterns based on insights"""
        
        predictions = []
        
        # Analyze insight patterns
        high_confidence_insights = [i for i in insights if i.confidence > 0.7]
        personality_insights = [i for i in insights if i.category == "personality"]
        
        if personality_insights:
            predictions.append("User's personality patterns suggest continued intellectual engagement")
        
        if user_model.trust_level > 0.6:
            predictions.append("Trust levels indicate user will likely deepen conversations")
        elif user_model.trust_level < 0.4:
            predictions.append("Low trust suggests user may remain cautious or test boundaries")
        
        if user_model.attachment_anxiety > 0.6:
            predictions.append("High attachment anxiety may lead to more validation-seeking behavior")
        
        return predictions
    
    async def _analyze_user_archetype_evolution(self, insights: List[ConversationInsight]) -> str:
        """Analyze how the user's archetype is evolving"""
        
        # Simple archetype analysis based on insight categories
        category_counts = {}
        for insight in insights:
            category_counts[insight.category] = category_counts.get(insight.category, 0) + 1
        
        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "unknown"
        
        archetype_mapping = {
            "personality": "The Evolving Self - exploring identity and expression",
            "desires": "The Seeker - driven by hidden wants and needs", 
            "fears": "The Vulnerable - managing anxieties and protection",
            "patterns": "The Habitual - defined by recurring behaviors",
            "communication": "The Expressive - focused on connection and understanding",
            "relationship": "The Relational - engaged in dynamic interaction"
        }
        
        return archetype_mapping.get(dominant_category, "The Mystery - still revealing themselves")

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Attempt to extract a top-level JSON object from arbitrary LLM text.

        Supports:
        - Plain JSON object
        - ```json ... ``` fenced blocks
        - ``` ... ``` fenced blocks
        - Prose with an embedded top-level JSON object (balanced braces)
        """
        if not text:
            return None

        s = text.strip()

        # 1) Exact JSON object
        if s.startswith('{'):
            try:
                json.loads(s)
                return s
            except Exception:
                pass

        # 2) Code-fenced JSON
        if s.startswith("```json"):
            try:
                inner = s.split("```json", 1)[1]
                inner = inner.split("```", 1)[0]
                candidate = inner.strip()
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        if s.startswith("```"):
            try:
                inner = s.split("```", 1)[1]
                inner = inner.split("```", 1)[0]
                candidate = inner.strip()
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        # 3) Scan for first balanced JSON object in the text
        first_brace = s.find('{')
        while first_brace != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(first_brace, len(s)):
                ch = s[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = s[first_brace:i+1].strip()
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                break
            first_brace = s.find('{', first_brace + 1)

        return None


# Global analyzer instance
post_conversation_analyzer = PostConversationAnalyzer()
