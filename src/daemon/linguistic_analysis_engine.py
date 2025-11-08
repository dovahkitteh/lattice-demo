"""
🩸 GLASSSHARD DAEMONCORE - Linguistic Analysis Engine
Deep linguistic pattern analysis and subtext detection for advanced user understanding
"""

import json
import re
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics

# NLP Libraries
import spacy
import textstat
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

logger = logging.getLogger(__name__)

class CommunicationStyle(Enum):
    """Communication style patterns"""
    FORMAL = "formal"
    CASUAL = "casual"
    EMOTIONAL = "emotional"
    LOGICAL = "logical"
    ASSERTIVE = "assertive"
    QUESTIONING = "questioning"
    DIRECT = "direct"
    INDIRECT = "indirect"
    COLLABORATIVE = "collaborative"
    INSTRUCTIONAL = "instructional"

class SubtextType(Enum):
    """Types of subtext detected"""
    IMPLICIT_REQUEST = "implicit_request"
    EMOTIONAL_UNDERTONE = "emotional_undertone"
    HIDDEN_CONCERN = "hidden_concern"
    VALIDATION_SEEKING = "validation_seeking"
    AUTHORITY_TESTING = "authority_testing"
    RELATIONSHIP_PROBING = "relationship_probing"
    INTELLECTUAL_CHALLENGE = "intellectual_challenge"
    VULNERABILITY_EXPRESSION = "vulnerability_expression"

class QuestionIntent(Enum):
    """Intent classification for questions"""
    INFORMATION_SEEKING = "information_seeking"
    VALIDATION_SEEKING = "validation_seeking"
    TESTING_KNOWLEDGE = "testing_knowledge"
    EXPLORATORY = "exploratory"
    RHETORICAL = "rhetorical"
    CLARIFICATION = "clarification"
    OPINION_SEEKING = "opinion_seeking"
    DECISION_SUPPORT = "decision_support"

@dataclass
class LinguisticPattern:
    """Individual linguistic pattern detected"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    evidence: List[str]
    context: str
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class SubtextDetection:
    """Detected subtext in communication"""
    subtext_id: str
    subtext_type: SubtextType
    surface_text: str
    implied_meaning: str
    confidence: float
    indicators: List[str]
    emotional_charge: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "subtext_id": self.subtext_id,
            "subtext_type": self.subtext_type.value,
            "surface_text": self.surface_text,
            "implied_meaning": self.implied_meaning,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "emotional_charge": self.emotional_charge,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class CommunicationProfile:
    """Communication style profile"""
    user_id: str
    dominant_styles: List[CommunicationStyle]
    style_distribution: Dict[str, float]
    complexity_level: float
    formality_level: float
    emotional_expressiveness: float
    questioning_tendency: float
    directness_level: float
    vocabulary_sophistication: float
    sentence_structure_complexity: float
    last_updated: datetime

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "dominant_styles": [style.value for style in self.dominant_styles],
            "style_distribution": self.style_distribution,
            "complexity_level": self.complexity_level,
            "formality_level": self.formality_level,
            "emotional_expressiveness": self.emotional_expressiveness,
            "questioning_tendency": self.questioning_tendency,
            "directness_level": self.directness_level,
            "vocabulary_sophistication": self.vocabulary_sophistication,
            "sentence_structure_complexity": self.sentence_structure_complexity,
            "last_updated": self.last_updated.isoformat()
        }

@dataclass
class SemanticRelationship:
    """Semantic relationship between concepts"""
    relationship_id: str
    concept_a: str
    concept_b: str
    relationship_type: str
    strength: float
    context: str
    frequency: int
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "relationship_id": self.relationship_id,
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "context": self.context,
            "frequency": self.frequency,
            "timestamp": self.timestamp.isoformat()
        }

class LinguisticAnalysisEngine:
    """
    Advanced linguistic analysis engine for deep user understanding
    """
    
    def __init__(self, device: str = "cpu"):
        # Detect available device with fallback
        self.device = self._detect_available_device(device)
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.pattern_counter = 0
        self.subtext_counter = 0
        self.relationship_counter = 0
        
        # Pattern storage
        self.detected_patterns: Dict[str, LinguisticPattern] = {}
        self.detected_subtext: Dict[str, SubtextDetection] = {}
        self.semantic_relationships: Dict[str, SemanticRelationship] = {}
        self.communication_profiles: Dict[str, CommunicationProfile] = {}
        
        # Initialize async
        self._initialized = False
    
    def _detect_available_device(self, preferred_device: str) -> str:
        """Detect the best available device with fallback"""
        try:
            import torch
            
            if preferred_device == "cuda":
                try:
                    if torch.cuda.is_available():
                        # Test if CUDA actually works by creating a tensor
                        test_tensor = torch.tensor([1.0]).to("cuda")
                        logger.info(f"🔬 Using CUDA device: {torch.cuda.get_device_name(0)}")
                        return "cuda"
                    else:
                        logger.warning("🔬 CUDA requested but not available, falling back to CPU")
                        return "cpu"
                except Exception as cuda_error:
                    logger.warning(f"🔬 CUDA test failed ({cuda_error}), falling back to CPU")
                    return "cpu"
            else:
                logger.info("🔬 Using CPU device")
                return "cpu"
        except Exception as e:
            logger.warning(f"🔬 Error detecting device, falling back to CPU: {e}")
            return "cpu"
        
    async def initialize(self):
        """Initialize the linguistic analysis models"""
        if self._initialized:
            return
            
        try:
            logger.info("🔬 Initializing linguistic analysis engine...")
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            # Load spaCy model (download if not available)
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Load BERT model for semantic analysis with device handling
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with robust device handling
            try:
                # First, load the model on CPU
                self.model = AutoModel.from_pretrained(model_name)
                logger.info(f"🔬 Loaded semantic model from pretrained")
                
                # Then try to move it to the target device
                if self.device == "cuda":
                    try:
                        self.model = self.model.to("cuda")
                        logger.info(f"🔬 Moved model to CUDA device")
                    except Exception as cuda_error:
                        logger.warning(f"🔬 Failed to move model to CUDA ({cuda_error}), keeping on CPU")
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                else:
                    self.model = self.model.to("cpu")
                    logger.info(f"🔬 Model running on CPU")
                    
            except Exception as model_error:
                logger.error(f"🔬 Failed to load semantic model: {model_error}")
                # Continue without the semantic model - other analyses will still work
                self.model = None
            
            self._initialized = True
            logger.info("✅ Linguistic analysis engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing linguistic analysis engine: {e}")
            raise

    async def analyze_message(self, message: str, user_id: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive linguistic analysis of a message
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            analysis_results = {}
            
            # Syntactic complexity analysis
            analysis_results["syntactic_complexity"] = await self._analyze_syntactic_complexity(message)
            
            # Communication style analysis
            analysis_results["communication_style"] = await self._analyze_communication_style(message)
            
            # Subtext detection
            analysis_results["subtext"] = await self._detect_subtext(message, context)
            
            # Question intent classification
            analysis_results["question_intent"] = await self._classify_question_intent(message)
            
            # Semantic relationships
            analysis_results["semantic_relationships"] = await self._extract_semantic_relationships(message)
            
            # Emotional undertones (beyond surface emotions)
            analysis_results["emotional_undertones"] = await self._analyze_emotional_undertones(message)
            
            # Metaphor and analogy detection
            analysis_results["metaphors_analogies"] = await self._detect_metaphors_analogies(message)
            
            # Update communication profile
            await self._update_communication_profile(user_id, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in linguistic analysis: {e}")
            return {}

    async def _analyze_syntactic_complexity(self, message: str) -> Dict[str, float]:
        """Analyze syntactic complexity of the message"""
        try:
            # Basic readability metrics
            flesch_reading_ease = textstat.flesch_reading_ease(message)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(message)
            gunning_fog = textstat.gunning_fog(message)
            
            # spaCy analysis
            doc = self.nlp(message)
            
            # Sentence complexity
            sentences = list(doc.sents)
            avg_sentence_length = len(doc) / max(len(sentences), 1)
            
            # Syntactic depth
            syntactic_depths = []
            for sent in sentences:
                depths = []
                for token in sent:
                    depth = self._calculate_syntactic_depth(token)
                    depths.append(depth)
                if depths:
                    syntactic_depths.append(max(depths))
            
            avg_syntactic_depth = statistics.mean(syntactic_depths) if syntactic_depths else 0
            
            # Part of speech diversity
            pos_counts = Counter([token.pos_ for token in doc])
            pos_diversity = len(pos_counts) / max(len(doc), 1)
            
            # Dependency complexity
            dep_counts = Counter([token.dep_ for token in doc])
            dep_diversity = len(dep_counts) / max(len(doc), 1)
            
            return {
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "gunning_fog": gunning_fog,
                "avg_sentence_length": avg_sentence_length,
                "avg_syntactic_depth": avg_syntactic_depth,
                "pos_diversity": pos_diversity,
                "dep_diversity": dep_diversity,
                "overall_complexity": (avg_syntactic_depth + pos_diversity + dep_diversity) / 3
            }
            
        except Exception as e:
            logger.error(f"Error in syntactic complexity analysis: {e}")
            return {}

    def _calculate_syntactic_depth(self, token) -> int:
        """Calculate syntactic depth of a token"""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth

    async def _analyze_communication_style(self, message: str) -> Dict[str, Any]:
        """Analyze communication style patterns"""
        try:
            doc = self.nlp(message)
            style_scores = {}
            
            # Formality indicators
            formal_words = {'furthermore', 'nevertheless', 'consequently', 'accordingly', 
                           'therefore', 'however', 'moreover', 'nonetheless'}
            casual_words = {'yeah', 'ok', 'okay', 'cool', 'awesome', 'kinda', 'sorta', 'btw'}
            
            formal_count = sum(1 for token in doc if token.text.lower() in formal_words)
            casual_count = sum(1 for token in doc if token.text.lower() in casual_words)
            
            style_scores['formality'] = (formal_count - casual_count) / max(len(doc), 1)
            
            # Emotional expressiveness
            emotional_words = {'love', 'hate', 'excited', 'disappointed', 'frustrated', 'amazing', 'terrible'}
            emotional_count = sum(1 for token in doc if token.text.lower() in emotional_words)
            exclamation_count = message.count('!')
            caps_count = sum(1 for char in message if char.isupper())
            
            style_scores['emotional_expressiveness'] = (emotional_count + exclamation_count + caps_count/100) / max(len(doc), 1)
            
            # Questioning tendency
            question_count = message.count('?')
            question_words = {'what', 'when', 'where', 'why', 'how', 'who', 'which'}
            question_word_count = sum(1 for token in doc if token.text.lower() in question_words)
            
            style_scores['questioning_tendency'] = (question_count + question_word_count) / max(len(doc), 1)
            
            # Directness
            imperative_count = sum(1 for sent in doc.sents if self._is_imperative(sent))
            direct_words = {'need', 'want', 'must', 'should', 'will', 'do', 'make', 'get'}
            direct_count = sum(1 for token in doc if token.text.lower() in direct_words)
            
            style_scores['directness'] = (imperative_count + direct_count) / max(len(doc), 1)
            
            # Logical structure
            logical_connectors = {'because', 'since', 'therefore', 'thus', 'so', 'if', 'then', 'when'}
            logical_count = sum(1 for token in doc if token.text.lower() in logical_connectors)
            
            style_scores['logical_structure'] = logical_count / max(len(doc), 1)
            
            # Determine dominant style
            dominant_styles = []
            if style_scores['formality'] > 0.1:
                dominant_styles.append(CommunicationStyle.FORMAL)
            elif style_scores['formality'] < -0.1:
                dominant_styles.append(CommunicationStyle.CASUAL)
                
            if style_scores['emotional_expressiveness'] > 0.2:
                dominant_styles.append(CommunicationStyle.EMOTIONAL)
            if style_scores['logical_structure'] > 0.1:
                dominant_styles.append(CommunicationStyle.LOGICAL)
            if style_scores['questioning_tendency'] > 0.2:
                dominant_styles.append(CommunicationStyle.QUESTIONING)
            if style_scores['directness'] > 0.2:
                dominant_styles.append(CommunicationStyle.DIRECT)
            
            return {
                "style_scores": style_scores,
                "dominant_styles": [style.value for style in dominant_styles],
                "overall_complexity": sum(style_scores.values()) / len(style_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in communication style analysis: {e}")
            return {}

    def _is_imperative(self, sent) -> bool:
        """Check if a sentence is imperative"""
        # Simple heuristic: starts with verb, no subject pronoun
        tokens = [token for token in sent if not token.is_punct]
        if not tokens:
            return False
        
        first_token = tokens[0]
        return (first_token.pos_ == "VERB" and 
                first_token.tag_ in ["VB", "VBP"] and
                not any(token.pos_ == "PRON" and token.dep_ == "nsubj" for token in tokens))

    async def _detect_subtext(self, message: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect subtext and implied meanings"""
        try:
            doc = self.nlp(message)
            subtext_detections = []
            
            # Implicit request detection
            implicit_requests = self._detect_implicit_requests(message, doc)
            subtext_detections.extend(implicit_requests)
            
            # Emotional undertones
            emotional_undertones = self._detect_emotional_undertones(message, doc)
            subtext_detections.extend(emotional_undertones)
            
            # Hidden concerns
            hidden_concerns = self._detect_hidden_concerns(message, doc)
            subtext_detections.extend(hidden_concerns)
            
            # Validation seeking
            validation_seeking = self._detect_validation_seeking(message, doc)
            subtext_detections.extend(validation_seeking)
            
            # Authority testing
            authority_testing = self._detect_authority_testing(message, doc)
            subtext_detections.extend(authority_testing)
            
            return subtext_detections
            
        except Exception as e:
            logger.error(f"Error in subtext detection: {e}")
            return []

    def _detect_implicit_requests(self, message: str, doc) -> List[Dict[str, Any]]:
        """Detect implicit requests in the message"""
        detections = []
        
        # Pattern: "I wonder if..." / "I'm curious about..."
        wonder_patterns = [
            r"i wonder if",
            r"i'm curious about",
            r"i'd like to know",
            r"it would be helpful if",
            r"it might be good to"
        ]
        
        for pattern in wonder_patterns:
            if re.search(pattern, message.lower()):
                detections.append({
                    "subtext_type": SubtextType.IMPLICIT_REQUEST.value,
                    "surface_text": message,
                    "implied_meaning": "User is making an indirect request for information or action",
                    "confidence": 0.8,
                    "indicators": [pattern],
                    "emotional_charge": 0.3
                })
        
        return detections

    def _detect_emotional_undertones(self, message: str, doc) -> List[Dict[str, Any]]:
        """Detect emotional undertones beyond surface emotions"""
        detections = []
        
        # Frustration indicators
        frustration_patterns = [
            r"still don't",
            r"keep having",
            r"always",
            r"never works",
            r"doesn't make sense"
        ]
        
        for pattern in frustration_patterns:
            if re.search(pattern, message.lower()):
                detections.append({
                    "subtext_type": SubtextType.EMOTIONAL_UNDERTONE.value,
                    "surface_text": message,
                    "implied_meaning": "User may be experiencing frustration or impatience",
                    "confidence": 0.7,
                    "indicators": [pattern],
                    "emotional_charge": 0.6
                })
        
        return detections

    def _detect_hidden_concerns(self, message: str, doc) -> List[Dict[str, Any]]:
        """Detect hidden concerns or anxieties"""
        detections = []
        
        concern_patterns = [
            r"what if",
            r"i hope",
            r"i'm worried",
            r"i don't want",
            r"hopefully"
        ]
        
        for pattern in concern_patterns:
            if re.search(pattern, message.lower()):
                detections.append({
                    "subtext_type": SubtextType.HIDDEN_CONCERN.value,
                    "surface_text": message,
                    "implied_meaning": "User may have underlying concerns or anxieties",
                    "confidence": 0.6,
                    "indicators": [pattern],
                    "emotional_charge": 0.4
                })
        
        return detections

    def _detect_validation_seeking(self, message: str, doc) -> List[Dict[str, Any]]:
        """Detect validation seeking behavior"""
        detections = []
        
        validation_patterns = [
            r"is that right",
            r"does that make sense",
            r"am i wrong",
            r"what do you think",
            r"do you agree"
        ]
        
        for pattern in validation_patterns:
            if re.search(pattern, message.lower()):
                detections.append({
                    "subtext_type": SubtextType.VALIDATION_SEEKING.value,
                    "surface_text": message,
                    "implied_meaning": "User is seeking validation or confirmation",
                    "confidence": 0.8,
                    "indicators": [pattern],
                    "emotional_charge": 0.5
                })
        
        return detections

    def _detect_authority_testing(self, message: str, doc) -> List[Dict[str, Any]]:
        """Detect authority testing behavior"""
        detections = []
        
        authority_patterns = [
            r"are you sure",
            r"how do you know",
            r"prove it",
            r"can you really",
            r"i doubt"
        ]
        
        for pattern in authority_patterns:
            if re.search(pattern, message.lower()):
                detections.append({
                    "subtext_type": SubtextType.AUTHORITY_TESTING.value,
                    "surface_text": message,
                    "implied_meaning": "User is testing or challenging authority/expertise",
                    "confidence": 0.7,
                    "indicators": [pattern],
                    "emotional_charge": 0.6
                })
        
        return detections

    async def _classify_question_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Classify the intent of questions in the message"""
        if '?' not in message:
            return None
            
        try:
            doc = self.nlp(message)
            
            # Extract questions
            questions = [sent.text for sent in doc.sents if '?' in sent.text]
            if not questions:
                return None
            
            intent_classification = {}
            
            for question in questions:
                question_lower = question.lower()
                
                # Information seeking
                if any(word in question_lower for word in ['what', 'when', 'where', 'who', 'which', 'how much']):
                    intent_classification[question] = {
                        "intent": QuestionIntent.INFORMATION_SEEKING.value,
                        "confidence": 0.8
                    }
                
                # Validation seeking
                elif any(phrase in question_lower for phrase in ['is that right', 'correct', 'do you think', 'agree']):
                    intent_classification[question] = {
                        "intent": QuestionIntent.VALIDATION_SEEKING.value,
                        "confidence": 0.7
                    }
                
                # Testing knowledge
                elif any(phrase in question_lower for phrase in ['do you know', 'can you tell me', 'are you familiar']):
                    intent_classification[question] = {
                        "intent": QuestionIntent.TESTING_KNOWLEDGE.value,
                        "confidence": 0.6
                    }
                
                # Exploratory
                elif any(word in question_lower for word in ['why', 'how']):
                    intent_classification[question] = {
                        "intent": QuestionIntent.EXPLORATORY.value,
                        "confidence": 0.7
                    }
                
                # Clarification
                elif any(phrase in question_lower for phrase in ['what do you mean', 'can you clarify', 'explain']):
                    intent_classification[question] = {
                        "intent": QuestionIntent.CLARIFICATION.value,
                        "confidence": 0.8
                    }
                
                # Default to exploratory
                else:
                    intent_classification[question] = {
                        "intent": QuestionIntent.EXPLORATORY.value,
                        "confidence": 0.5
                    }
            
            return intent_classification
            
        except Exception as e:
            logger.error(f"Error in question intent classification: {e}")
            return None

    async def _extract_semantic_relationships(self, message: str) -> List[Dict[str, Any]]:
        """Extract semantic relationships between concepts"""
        try:
            doc = self.nlp(message)
            relationships = []
            
            # Extract noun phrases and named entities
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            entities = [ent.text for ent in doc.ents]
            
            # Combine concepts
            concepts = list(set(noun_phrases + entities))
            
            # Find relationships through dependency parsing
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                        head = token.head
                        if head.pos_ == 'VERB':
                            relationship = {
                                "concept_a": token.text,
                                "concept_b": head.text,
                                "relationship_type": token.dep_,
                                "strength": 0.7,
                                "context": sent.text
                            }
                            relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in semantic relationship extraction: {e}")
            return []

    async def _analyze_emotional_undertones(self, message: str) -> Dict[str, float]:
        """Analyze emotional undertones beyond surface emotions"""
        try:
            # Use VADER sentiment for baseline
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            
            sentiment_scores = sia.polarity_scores(message)
            
            # Additional emotional undertone analysis
            undertones = {
                "anxiety": self._detect_anxiety_markers(message),
                "confidence": self._detect_confidence_markers(message),
                "curiosity": self._detect_curiosity_markers(message),
                "frustration": self._detect_frustration_markers(message),
                "enthusiasm": self._detect_enthusiasm_markers(message),
                "uncertainty": self._detect_uncertainty_markers(message)
            }
            
            # Combine with sentiment scores
            undertones.update(sentiment_scores)
            
            return undertones
            
        except Exception as e:
            logger.error(f"Error in emotional undertone analysis: {e}")
            return {}

    def _detect_anxiety_markers(self, message: str) -> float:
        """Detect anxiety markers in text"""
        anxiety_words = ['worried', 'nervous', 'anxious', 'concerned', 'afraid', 'scared', 'stress']
        anxiety_phrases = ['what if', 'i hope', 'hopefully', 'i don\'t want']
        
        word_count = sum(1 for word in anxiety_words if word in message.lower())
        phrase_count = sum(1 for phrase in anxiety_phrases if phrase in message.lower())
        
        return (word_count + phrase_count) / max(len(message.split()), 1)

    def _detect_confidence_markers(self, message: str) -> float:
        """Detect confidence markers in text"""
        confidence_words = ['sure', 'certain', 'confident', 'definitely', 'absolutely', 'clearly', 'obviously']
        confidence_phrases = ['i know', 'i\'m sure', 'without doubt', 'no question']
        
        word_count = sum(1 for word in confidence_words if word in message.lower())
        phrase_count = sum(1 for phrase in confidence_phrases if phrase in message.lower())
        
        return (word_count + phrase_count) / max(len(message.split()), 1)

    def _detect_curiosity_markers(self, message: str) -> float:
        """Detect curiosity markers in text"""
        curiosity_words = ['curious', 'wonder', 'interesting', 'fascinating', 'explore', 'discover']
        question_count = message.count('?')
        
        word_count = sum(1 for word in curiosity_words if word in message.lower())
        
        return (word_count + question_count * 0.5) / max(len(message.split()), 1)

    def _detect_frustration_markers(self, message: str) -> float:
        """Detect frustration markers in text"""
        frustration_words = ['frustrated', 'annoyed', 'irritated', 'stuck', 'difficult', 'hard']
        frustration_phrases = ['doesn\'t work', 'keep having', 'always', 'never works']
        
        word_count = sum(1 for word in frustration_words if word in message.lower())
        phrase_count = sum(1 for phrase in frustration_phrases if phrase in message.lower())
        
        return (word_count + phrase_count) / max(len(message.split()), 1)

    def _detect_enthusiasm_markers(self, message: str) -> float:
        """Detect enthusiasm markers in text"""
        enthusiasm_words = ['excited', 'amazing', 'awesome', 'fantastic', 'great', 'wonderful', 'love']
        exclamation_count = message.count('!')
        caps_words = sum(1 for word in message.split() if word.isupper() and len(word) > 2)
        
        word_count = sum(1 for word in enthusiasm_words if word in message.lower())
        
        return (word_count + exclamation_count * 0.3 + caps_words * 0.2) / max(len(message.split()), 1)

    def _detect_uncertainty_markers(self, message: str) -> float:
        """Detect uncertainty markers in text"""
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems', 'appears']
        uncertainty_phrases = ['i think', 'i believe', 'i guess', 'not sure', 'i don\'t know']
        
        word_count = sum(1 for word in uncertainty_words if word in message.lower())
        phrase_count = sum(1 for phrase in uncertainty_phrases if phrase in message.lower())
        
        return (word_count + phrase_count) / max(len(message.split()), 1)

    async def _detect_metaphors_analogies(self, message: str) -> List[Dict[str, Any]]:
        """Detect metaphors and analogies in the message"""
        try:
            metaphors_analogies = []
            
            # Common metaphor patterns
            metaphor_patterns = [
                r"like a",
                r"as if",
                r"reminds me of",
                r"similar to",
                r"just like",
                r"kind of like"
            ]
            
            for pattern in metaphor_patterns:
                matches = re.finditer(pattern, message.lower())
                for match in matches:
                    context = message[max(0, match.start()-20):match.end()+20]
                    metaphors_analogies.append({
                        "type": "metaphor_analogy",
                        "pattern": pattern,
                        "context": context,
                        "confidence": 0.7
                    })
            
            return metaphors_analogies
            
        except Exception as e:
            logger.error(f"Error in metaphor/analogy detection: {e}")
            return []

    async def _update_communication_profile(self, user_id: str, analysis_results: Dict[str, Any]):
        """Update the communication profile for a user"""
        try:
            if user_id not in self.communication_profiles:
                # Create new profile
                self.communication_profiles[user_id] = CommunicationProfile(
                    user_id=user_id,
                    dominant_styles=[],
                    style_distribution={},
                    complexity_level=0.0,
                    formality_level=0.0,
                    emotional_expressiveness=0.0,
                    questioning_tendency=0.0,
                    directness_level=0.0,
                    vocabulary_sophistication=0.0,
                    sentence_structure_complexity=0.0,
                    last_updated=datetime.now(timezone.utc)
                )
            
            profile = self.communication_profiles[user_id]
            
            # Update with new analysis results
            if "communication_style" in analysis_results:
                style_data = analysis_results["communication_style"]
                
                # Update style scores with smoothing
                alpha = 0.3  # Learning rate
                for style, score in style_data.get("style_scores", {}).items():
                    if style in profile.style_distribution:
                        profile.style_distribution[style] = (
                            profile.style_distribution[style] * (1 - alpha) + 
                            score * alpha
                        )
                    else:
                        profile.style_distribution[style] = score
                
                # Update complexity level
                if "overall_complexity" in style_data:
                    profile.complexity_level = (
                        profile.complexity_level * (1 - alpha) + 
                        style_data["overall_complexity"] * alpha
                    )
            
            if "syntactic_complexity" in analysis_results:
                complexity_data = analysis_results["syntactic_complexity"]
                
                # Update vocabulary sophistication
                if "overall_complexity" in complexity_data:
                    profile.vocabulary_sophistication = (
                        profile.vocabulary_sophistication * (1 - alpha) + 
                        complexity_data["overall_complexity"] * alpha
                    )
                
                # Update sentence structure complexity
                if "avg_syntactic_depth" in complexity_data:
                    profile.sentence_structure_complexity = (
                        profile.sentence_structure_complexity * (1 - alpha) + 
                        complexity_data["avg_syntactic_depth"] * alpha
                    )
            
            # Update other metrics
            if "emotional_undertones" in analysis_results:
                undertones = analysis_results["emotional_undertones"]
                if "enthusiasm" in undertones:
                    profile.emotional_expressiveness = (
                        profile.emotional_expressiveness * (1 - alpha) + 
                        undertones["enthusiasm"] * alpha
                    )
            
            profile.last_updated = datetime.now(timezone.utc)
            
            # Determine dominant styles
            if profile.style_distribution:
                sorted_styles = sorted(profile.style_distribution.items(), key=lambda x: x[1], reverse=True)
                profile.dominant_styles = [
                    CommunicationStyle(style) for style, score in sorted_styles[:3] 
                    if score > 0.2
                ]
            
            logger.debug(f"Updated communication profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating communication profile: {e}")

    def get_communication_profile(self, user_id: str) -> Optional[CommunicationProfile]:
        """Get communication profile for a user"""
        return self.communication_profiles.get(user_id)

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis engine state"""
        return {
            "total_patterns_detected": len(self.detected_patterns),
            "total_subtext_detected": len(self.detected_subtext),
            "total_semantic_relationships": len(self.semantic_relationships),
            "total_communication_profiles": len(self.communication_profiles),
            "initialized": self._initialized
        } 