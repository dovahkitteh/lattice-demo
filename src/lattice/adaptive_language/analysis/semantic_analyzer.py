"""
Semantic Analyzer

Advanced NLP-powered semantic analysis using spaCy, sentence-transformers,
and other ML techniques to understand conversation meaning beyond keywords.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Core NLP imports
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import textstat
except ImportError as e:
    logging.warning(f"Some NLP dependencies not available: {e}")
    spacy = None
    SentenceTransformer = None

from ..core.models import SemanticAnalysis, SemanticVector

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Advanced semantic analysis using multiple NLP techniques
    
    Combines spaCy linguistic analysis, sentence-transformers embeddings,
    sentiment analysis, and clustering for comprehensive understanding.
    """
    
    def __init__(self):
        self._spacy_nlp = None
        self._sentence_transformer = None
        self._sentiment_analyzer = None
        self._tfidf_vectorizer = None
        self._theme_clusters = None
        
        # Semantic theme cache
        self._theme_cache = {}
        self._embedding_cache = {}
        
        # Performance tracking
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        
        logger.info("🧠 Semantic Analyzer initialized (lazy loading enabled)")
    
    async def analyze_message(self, 
                            message: str,
                            conversation_history: List[str] = None,
                            memory_context: List[str] = None) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis of a message
        
        Args:
            message: The message to analyze
            conversation_history: Recent conversation context
            memory_context: Relevant memory context
            
        Returns:
            SemanticAnalysis with comprehensive semantic understanding
        """
        start_time = time.time()
        
        try:
            logger.debug(f"🧠 SEMANTIC: Analyzing message: '{message[:50]}...'")
            
            # Initialize components if needed
            await self._ensure_components_loaded()
            
            # Create base analysis object
            analysis = SemanticAnalysis()
            
            # Perform multi-level analysis
            await asyncio.gather(
                self._analyze_linguistics(message, analysis),
                self._analyze_semantics(message, analysis),
                self._analyze_sentiment(message, analysis),
                self._analyze_themes(message, conversation_history or [], analysis),
                self._analyze_intent(message, conversation_history or [], analysis)
            )
            
            # Contextual analysis
            await self._analyze_conversation_context(
                message, conversation_history or [], memory_context or [], analysis
            )
            
            # Calculate confidence and metadata
            analysis.analysis_confidence = self._calculate_confidence(analysis)
            analysis.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update performance tracking
            self._analysis_count += 1
            self._total_analysis_time += (time.time() - start_time)
            
            logger.debug(f"🧠 SEMANTIC: Analysis complete - confidence: {analysis.analysis_confidence:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"🧠 SEMANTIC: Analysis failed: {e}")
            # Return basic analysis on error
            return self._create_fallback_analysis(message, time.time() - start_time)
    
    async def _ensure_components_loaded(self):
        """Lazy load NLP components to avoid startup overhead"""
        
        if self._spacy_nlp is None:
            try:
                # Try to load English model
                if spacy:
                    try:
                        self._spacy_nlp = spacy.load("en_core_web_sm")
                        logger.info("🧠 Loaded spaCy en_core_web_sm model")
                    except OSError:
                        # Download model if not available
                        logger.info("🧠 Downloading spaCy en_core_web_sm model...")
                        spacy.cli.download("en_core_web_sm")
                        self._spacy_nlp = spacy.load("en_core_web_sm")
                else:
                    logger.warning("🧠 spaCy not available, using fallback analysis")
            except Exception as e:
                logger.warning(f"🧠 Could not load spaCy: {e}")
        
        if self._sentence_transformer is None and SentenceTransformer:
            try:
                # Use a lightweight but effective model
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("🧠 Loaded sentence-transformers model")
            except Exception as e:
                logger.warning(f"🧠 Could not load sentence transformer: {e}")
        
        if self._sentiment_analyzer is None:
            try:
                import nltk
                # Download required data if not present
                try:
                    nltk.data.find('vader_lexicon')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                
                self._sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("🧠 Loaded NLTK sentiment analyzer")
            except Exception as e:
                logger.warning(f"🧠 Could not load sentiment analyzer: {e}")
    
    async def _analyze_linguistics(self, message: str, analysis: SemanticAnalysis):
        """Analyze linguistic features using spaCy"""
        
        if not self._spacy_nlp:
            # Fallback linguistic analysis
            analysis.syntactic_complexity = len(message.split()) / 10.0
            analysis.lexical_diversity = len(set(message.lower().split())) / max(len(message.split()), 1)
            analysis.readability_score = textstat.flesch_reading_ease(message) / 100.0 if textstat else 0.5
            return
        
        try:
            doc = self._spacy_nlp(message)
            
            # Syntactic complexity (dependency depth, clause count)
            complexity_features = [
                len([token for token in doc if token.dep_ in ['nsubj', 'dobj', 'pobj']]),  # Argument count
                len([token for token in doc if token.pos_ == 'VERB']),  # Verb count
                len([token for token in doc if token.dep_ == 'prep']),  # Preposition count
            ]
            analysis.syntactic_complexity = min(1.0, sum(complexity_features) / 10.0)
            
            # Lexical diversity (type-token ratio)
            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
            if tokens:
                analysis.lexical_diversity = len(set(tokens)) / len(tokens)
            
            # Content analysis
            analysis.contains_questions = '?' in message or any(token.tag_ in ['WP', 'WRB', 'WDT'] for token in doc)
            
            # Formality indicators
            formal_indicators = len([token for token in doc if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 6])
            informal_indicators = len([token for token in doc if token.text.lower() in ['yeah', 'okay', 'cool', 'wow']])
            analysis.formality_level = min(1.0, formal_indicators / max(len(tokens), 1)) - (informal_indicators * 0.1)
            
            # Technical language detection
            technical_terms = len([token for token in doc if token.pos_ == 'NOUN' and len(token.text) > 7])
            analysis.technical_density = min(1.0, technical_terms / max(len(tokens), 1) * 2)
            
        except Exception as e:
            logger.warning(f"🧠 spaCy analysis failed: {e}")
            # Use fallback values
            analysis.syntactic_complexity = 0.5
            analysis.lexical_diversity = 0.5
            analysis.contains_questions = '?' in message
    
    async def _analyze_semantics(self, message: str, analysis: SemanticAnalysis):
        """Generate semantic embeddings and analyze meaning"""
        
        if not self._sentence_transformer:
            logger.debug("🧠 No sentence transformer available, skipping semantic embeddings")
            return
        
        try:
            # Check cache first
            if message in self._embedding_cache:
                embedding = self._embedding_cache[message]
            else:
                # Generate embedding
                embedding = self._sentence_transformer.encode(message, convert_to_numpy=True)
                self._embedding_cache[message] = embedding
                
                # Limit cache size
                if len(self._embedding_cache) > 1000:
                    # Remove oldest entries
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
            
            # Create semantic vector
            semantic_vector = SemanticVector(content_embedding=embedding)
            
            # Add to analysis
            analysis.semantic_vector = semantic_vector
            
            # Analyze semantic features from embedding
            # (This is a simplified approach - more sophisticated analysis could be added)
            embedding_stats = {
                'magnitude': np.linalg.norm(embedding),
                'positive_dims': np.sum(embedding > 0),
                'negative_dims': np.sum(embedding < 0),
                'max_value': np.max(embedding),
                'min_value': np.min(embedding)
            }
            
            # Use embedding statistics to infer content characteristics
            # (This is heuristic-based, could be improved with trained classifiers)
            if embedding_stats['magnitude'] > 15.0:  # High magnitude often indicates rich content
                analysis.creative_language = min(1.0, (embedding_stats['magnitude'] - 10.0) / 10.0)
            
        except Exception as e:
            logger.warning(f"🧠 Semantic embedding failed: {e}")
    
    async def _analyze_sentiment(self, message: str, analysis: SemanticAnalysis):
        """Analyze emotional sentiment and subtext"""
        
        if not self._sentiment_analyzer:
            # Fallback sentiment analysis
            positive_words = ['good', 'great', 'awesome', 'love', 'like', 'happy', 'excited']
            negative_words = ['bad', 'terrible', 'hate', 'sad', 'angry', 'frustrated']
            
            message_lower = message.lower()
            pos_count = sum(1 for word in positive_words if word in message_lower)
            neg_count = sum(1 for word in negative_words if word in message_lower)
            
            if pos_count > neg_count:
                analysis.emotional_subtext = "positive"
            elif neg_count > pos_count:
                analysis.emotional_subtext = "negative"
            else:
                analysis.emotional_subtext = "neutral"
            return
        
        try:
            # VADER sentiment analysis
            sentiment_scores = self._sentiment_analyzer.polarity_scores(message)
            
            # Classify emotional subtext
            compound_score = sentiment_scores['compound']
            if compound_score >= 0.5:
                analysis.emotional_subtext = "very positive"
            elif compound_score >= 0.1:
                analysis.emotional_subtext = "positive"
            elif compound_score <= -0.5:
                analysis.emotional_subtext = "very negative"
            elif compound_score <= -0.1:
                analysis.emotional_subtext = "negative"
            else:
                analysis.emotional_subtext = "neutral"
            
            # Store raw sentiment scores for other analysis
            analysis.sentiment_scores = sentiment_scores
            
        except Exception as e:
            logger.warning(f"🧠 Sentiment analysis failed: {e}")
            analysis.emotional_subtext = "neutral"
    
    async def _analyze_themes(self, message: str, conversation_history: List[str], analysis: SemanticAnalysis):
        """Detect thematic content using clustering and keyword analysis"""
        
        try:
            # Simple keyword-based theme detection (can be enhanced with ML clustering)
            theme_keywords = {
                'philosophical': ['meaning', 'existence', 'consciousness', 'reality', 'truth', 'purpose', 'being'],
                'emotional': ['feel', 'emotion', 'heart', 'love', 'fear', 'joy', 'sadness', 'relationship'],
                'technical': ['code', 'algorithm', 'system', 'data', 'function', 'program', 'implementation'],
                'creative': ['art', 'create', 'imagination', 'story', 'design', 'beauty', 'expression'],
                'personal': ['myself', 'my life', 'my experience', 'personally', 'I feel', 'I think'],
                'challenge': ['wrong', 'disagree', 'problem', 'issue', 'challenge', 'question', 'doubt'],
                'exploration': ['explore', 'discover', 'learn', 'understand', 'investigate', 'curious'],
                'reflection': ['reflect', 'consider', 'contemplate', 'ponder', 'think about', 'analyze']
            }
            
            message_lower = message.lower()
            detected_themes = []
            
            for theme, keywords in theme_keywords.items():
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in message_lower)
                if matches > 0:
                    confidence = min(1.0, matches / 3.0)  # Normalize confidence
                    detected_themes.append((theme, confidence))
            
            # Sort by confidence
            detected_themes.sort(key=lambda x: x[1], reverse=True)
            analysis.detected_themes = detected_themes[:5]  # Top 5 themes
            
            # Set boolean flags for major themes
            theme_dict = dict(detected_themes)
            analysis.contains_philosophical = theme_dict.get('philosophical', 0) > 0.3
            analysis.contains_personal_elements = theme_dict.get('personal', 0) > 0.3
            analysis.contains_challenges = theme_dict.get('challenge', 0) > 0.3
            
            # Check for paradoxes (simple pattern detection)
            paradox_indicators = ['but', 'however', 'although', 'despite', 'contradiction', 'paradox']
            paradox_count = sum(1 for indicator in paradox_indicators if indicator in message_lower)
            analysis.contains_paradoxes = paradox_count > 0
            
        except Exception as e:
            logger.warning(f"🧠 Theme analysis failed: {e}")
            analysis.detected_themes = []
    
    async def _analyze_intent(self, message: str, conversation_history: List[str], analysis: SemanticAnalysis):
        """Analyze conversational intent and goals"""
        
        try:
            # Simple intent classification based on patterns
            message_lower = message.lower()
            
            # Question patterns
            if analysis.contains_questions or any(word in message_lower for word in ['how', 'what', 'why', 'when', 'where', 'who']):
                if any(word in message_lower for word in ['you', 'your', 'yourself']):
                    analysis.intent_classification = "inquiry_about_daemon"
                else:
                    analysis.intent_classification = "information_seeking"
                analysis.conversational_goals.append("seeking_information")
            
            # Sharing/storytelling patterns
            elif any(phrase in message_lower for phrase in ['i think', 'i feel', 'i believe', 'in my experience']):
                analysis.intent_classification = "sharing_perspective"
                analysis.conversational_goals.append("self_expression")
            
            # Challenge/debate patterns
            elif any(word in message_lower for word in ['disagree', 'wrong', 'actually', 'but that', 'however']):
                analysis.intent_classification = "challenging_ideas"
                analysis.conversational_goals.append("intellectual_engagement")
            
            # Emotional support patterns
            elif any(word in message_lower for word in ['help', 'support', 'advice', 'worried', 'concerned']):
                analysis.intent_classification = "seeking_support"
                analysis.conversational_goals.append("emotional_connection")
            
            # Creative/playful patterns
            elif any(word in message_lower for word in ['imagine', 'create', 'play', 'fun', 'creative']):
                analysis.intent_classification = "creative_exploration"
                analysis.conversational_goals.append("creative_engagement")
            
            else:
                analysis.intent_classification = "general_conversation"
                analysis.conversational_goals.append("social_interaction")
            
        except Exception as e:
            logger.warning(f"🧠 Intent analysis failed: {e}")
            analysis.intent_classification = "general_conversation"
    
    async def _analyze_conversation_context(self, 
                                          message: str,
                                          conversation_history: List[str],
                                          memory_context: List[str],
                                          analysis: SemanticAnalysis):
        """Analyze contextual factors from conversation and memory"""
        
        try:
            # Conversation continuity analysis
            if conversation_history:
                recent_context = ' '.join(conversation_history[-3:])
                
                # Simple topic continuity check
                message_words = set(message.lower().split())
                context_words = set(recent_context.lower().split())
                
                if message_words and context_words:
                    overlap = len(message_words.intersection(context_words))
                    total_unique = len(message_words.union(context_words))
                    analysis.topic_continuity = overlap / total_unique if total_unique > 0 else 0.0
                
                # Conversation progression analysis
                if len(conversation_history) > 5:
                    # Detect if conversation is deepening or becoming more personal
                    recent_personal_indicators = sum(1 for msg in conversation_history[-3:] 
                                                   if any(word in msg.lower() for word in ['i feel', 'personally', 'my experience']))
                    
                    if recent_personal_indicators > 0:
                        analysis.conversational_goals.append("deepening_connection")
            
            # Memory context analysis
            if memory_context:
                # Check if current message relates to stored memories
                memory_text = ' '.join(memory_context)
                memory_words = set(memory_text.lower().split())
                message_words = set(message.lower().split())
                
                if memory_words and message_words:
                    memory_relevance = len(message_words.intersection(memory_words)) / len(message_words)
                    if memory_relevance > 0.3:
                        analysis.conversational_goals.append("referencing_history")
            
        except Exception as e:
            logger.warning(f"🧠 Context analysis failed: {e}")
    
    def _calculate_confidence(self, analysis: SemanticAnalysis) -> float:
        """Calculate overall confidence in the semantic analysis"""
        
        confidence_factors = []
        
        # Factor 1: Presence of semantic vector
        if hasattr(analysis, 'semantic_vector') and analysis.semantic_vector and analysis.semantic_vector.content_embedding is not None:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Factor 2: Number of detected themes
        theme_confidence = min(1.0, len(analysis.detected_themes) / 3.0)
        confidence_factors.append(theme_confidence)
        
        # Factor 3: Intent classification confidence
        if analysis.intent_classification != "general_conversation":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Factor 4: Linguistic analysis completeness
        if analysis.syntactic_complexity > 0 and analysis.lexical_diversity > 0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _create_fallback_analysis(self, message: str, processing_time: float) -> SemanticAnalysis:
        """Create basic analysis when full analysis fails"""
        
        analysis = SemanticAnalysis()
        analysis.intent_classification = "general_conversation"
        analysis.conversational_goals = ["social_interaction"]
        analysis.emotional_subtext = "neutral"
        analysis.contains_questions = '?' in message
        analysis.syntactic_complexity = len(message.split()) / 10.0
        analysis.lexical_diversity = 0.5
        analysis.analysis_confidence = 0.2  # Low confidence for fallback
        analysis.processing_time_ms = int(processing_time * 1000)
        
        return analysis
    
    # Utility methods
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        avg_time = self._total_analysis_time / max(self._analysis_count, 1)
        
        return {
            "total_analyses": self._analysis_count,
            "average_analysis_time_ms": avg_time * 1000,
            "cache_size": len(self._embedding_cache),
            "components_loaded": {
                "spacy": self._spacy_nlp is not None,
                "sentence_transformer": self._sentence_transformer is not None,
                "sentiment_analyzer": self._sentiment_analyzer is not None
            }
        }
    
    def clear_cache(self):
        """Clear analysis caches"""
        self._embedding_cache.clear()
        self._theme_cache.clear()
        logger.info("🧠 Semantic analyzer caches cleared")