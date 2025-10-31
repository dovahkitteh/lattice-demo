# src/lattice/emotions/memory_store.py
"""
Handles the persistence and retrieval of episodic traces related to the emotional state.
This includes:
- Storing a detailed record of each interaction turn.
- Retrieving relevant past episodes based on semantic and emotional similarity.
- Managing the retention policy for episodic traces.
"""
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from ..models import EpisodicTrace
from .. import config

logger = logging.getLogger(__name__)

EPISODIC_COLLECTION_NAME = "episodic_traces"

def get_episodic_collection():
    """Gets the ChromaDB collection for episodic traces (uses main memories collection with prefix)."""
    if not config.chroma_db:
        logger.error("ChromaDB client is not available.")
        raise RuntimeError("ChromaDB not initialized")
    
    logger.debug(f"Using main memories collection for episodic traces: {type(config.chroma_db)}")
    logger.debug(f"🔍 config.chroma_db has 'add' method: {hasattr(config.chroma_db, 'add')}")
    
    # Use the main memories collection - it's already working correctly
    # We'll store episodic traces with a "episodic_" prefix in their IDs
    if hasattr(config.chroma_db, 'add') and not hasattr(config.chroma_db, 'model_dump'):
        logger.debug(f"✅ Returning main memories collection for episodic storage: {type(config.chroma_db)}")
        return config.chroma_db
    else:
        logger.error(f"❌ Main memories collection is not a valid ChromaDB collection: {type(config.chroma_db)}")
        raise RuntimeError("Main memories collection is not accessible")

async def save_episodic_trace(trace: EpisodicTrace):
    """
    Saves an episodic trace to ChromaDB with emotional state embeddings.
    """
    try:

        logger.debug(f"🔍 Starting episodic trace save for turn_id: {trace.turn_id}")
        
        # Get the episodic collection (with debugging)
        collection = get_episodic_collection()
        logger.debug(f"🔍 Got collection from get_episodic_collection(): {type(collection)}")
        logger.debug(f"🔍 Collection has required methods - add: {hasattr(collection, 'add')}, count: {hasattr(collection, 'count')}")
        
        # Create a text representation of the trace for embedding
        document_to_embed = f"Turn {trace.turn_id}: {trace.mood_family} mood with intensity {trace.intensity:.2f}. "
        if trace.core_emotion:
            document_to_embed += f"Core emotion: {trace.core_emotion}. "
        if trace.distortion_type:
            document_to_embed += f"Distortion: {trace.distortion_type}. "
        if trace.applied_seeds:
            document_to_embed += f"Applied seeds: {', '.join(trace.applied_seeds)}. "
        document_to_embed += f"Context: User interaction in {trace.session_id}"
        
        logger.debug(f"Generated document for embedding: {document_to_embed[:100]}...")
        
        # Generate semantic embedding for the trace
        semantic_embedding = config.embedder.encode(document_to_embed).tolist()
        logger.debug(f"Generated semantic embedding with dimension: {len(semantic_embedding)}")
        
        # Create combined embedding (semantic + emotional) to match collection format
        # Use the raw_vector_post as the emotional component for episodic traces
        emotional_vector = trace.raw_vector_post if len(trace.raw_vector_post) == 28 else [0.0] * 28
        combined_embedding = semantic_embedding + emotional_vector
        logger.debug(f"Created combined embedding with dimension: {len(combined_embedding)} (semantic: {len(semantic_embedding)}, emotional: {len(emotional_vector)})")

        # Episodic traces need a unique ID for storage with episodic prefix
        trace_id = f"episodic_{uuid.uuid4()}"
        logger.debug(f"Generated episodic trace ID: {trace_id}")

        # Metadata will store the full trace object, minus fields already used.
        # Convert to dict and handle vector fields that ChromaDB can't store
        metadata = trace.model_dump()
        metadata['stored_at_utc'] = datetime.now(timezone.utc).isoformat()
        
        # Convert complex data structures to JSON strings for ChromaDB compatibility
        # Also handle None values since ChromaDB doesn't accept None
        if 'raw_vector_pre' in metadata:
            if metadata['raw_vector_pre']:
                metadata['raw_vector_pre'] = json.dumps(metadata['raw_vector_pre'])
            else:
                metadata['raw_vector_pre'] = "[]"  # Empty array as string
                
        if 'raw_vector_post' in metadata:
            if metadata['raw_vector_post']:
                metadata['raw_vector_post'] = json.dumps(metadata['raw_vector_post'])
            else:
                metadata['raw_vector_post'] = "[]"  # Empty array as string
                
        if 'dimension_snapshot' in metadata:
            if metadata['dimension_snapshot']:
                metadata['dimension_snapshot'] = json.dumps(metadata['dimension_snapshot'])
            else:
                metadata['dimension_snapshot'] = "{}"  # Empty object as string
                
        if 'param_modulation' in metadata:
            if metadata['param_modulation']:
                metadata['param_modulation'] = json.dumps(metadata['param_modulation'])
            else:
                metadata['param_modulation'] = "{}"  # Empty object as string
                
        if 'applied_seeds' in metadata:
            if metadata['applied_seeds']:
                metadata['applied_seeds'] = json.dumps(metadata['applied_seeds'])
            else:
                metadata['applied_seeds'] = "[]"  # Empty array as string
        
        # Handle other potential None values by converting them to appropriate defaults
        for key, value in metadata.items():
            if value is None:
                if key in ['distorted_meaning', 'interpretation_delta', 'mood_family', 'distortion_type', 'user_text', 'session_id', 'core_emotion']:
                    metadata[key] = ""  # Empty string for text fields
                elif key in ['turn_id']:
                    metadata[key] = 0  # Zero for numeric fields
                elif key in ['intensity']:
                    metadata[key] = 0.0  # Zero float for float fields
                else:
                    metadata[key] = ""  # Default to empty string for unknown fields
        
        logger.debug(f"Prepared metadata with keys: {list(metadata.keys())}")
        logger.debug(f"Vector fields converted to JSON strings for ChromaDB compatibility")
        
        # Debug the collection object before trying to use it
        logger.debug(f"🔍 Collection object type: {type(collection)}")
        logger.debug(f"🔍 Collection object dir: {[attr for attr in dir(collection) if 'add' in attr.lower()]}")
        logger.debug(f"🔍 Has 'add' method: {hasattr(collection, 'add')}")
        logger.debug(f"🔍 Collection repr: {repr(collection)}")
        
        # Check if this is a Pydantic model vs ChromaDB collection
        if hasattr(collection, 'model_dump'):
            logger.error("❌ Collection is a Pydantic model, not a ChromaDB collection!")
            logger.error(f"❌ Pydantic model fields: {list(collection.model_fields.keys()) if hasattr(collection, 'model_fields') else 'No model_fields'}")
            raise RuntimeError("Invalid collection object: got Pydantic model instead of ChromaDB collection")
        
        if not hasattr(collection, 'add'):
            logger.error(f"❌ Collection object missing 'add' method. Type: {type(collection)}")
            raise RuntimeError("Invalid collection object: missing 'add' method")
        
        logger.debug("✅ Collection object validation passed, calling add()...")
        
        collection.add(
            ids=[trace_id],
            embeddings=[combined_embedding],
            documents=[document_to_embed],
            metadatas=[metadata]
        )
        
        logger.info(f"✅ EPISODIC_TRACE_V2_STORED: turn_id={trace.turn_id}, collection_id={trace_id}, mood={trace.mood_family}")
        logger.debug(f"Collection now has {collection.count()} total items")

    except Exception as e:
        logger.error(f"❌ Failed to save episodic trace to ChromaDB: {e}", exc_info=True)
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception args: {e.args}")
        
        # Additional debugging for the specific Pydantic error
        if "'Collection' object has no attribute 'add'" in str(e):
            logger.error("❌ CONFIRMED: Received Pydantic Collection model instead of ChromaDB collection")
            logger.error("❌ This indicates an issue in get_episodic_collection() function")
            logger.error("❌ Check the collection retrieval and creation logic")


async def retrieve_episodic_traces_by_session(session_id: str, limit: int = 10) -> List[EpisodicTrace]:
    """
    Retrieves episodic traces for a specific session, ordered by turn_id (most recent first).
    
    Args:
        session_id: The session ID to filter by
        limit: Maximum number of traces to return
        
    Returns:
        List of EpisodicTrace objects for the session
    """
    if not config.embedder:
        logger.error("Embedder not initialized, cannot retrieve episodic traces.")
        return []

    try:
        collection = get_episodic_collection()
        
        # Use query method to get items by session, then filter by metadata to identify episodic traces
        # We'll use a dummy query vector and rely on metadata filtering
        dummy_query = [0.0] * (config.EMBED_DIM + 28)
        results = collection.query(
            query_embeddings=[dummy_query],
            n_results=1000,  # Get lots to ensure we capture all for this session
            where={"session_id": session_id},
            include=["metadatas"]
        )
        
        # Filter for actual episodic traces by checking metadata structure
        filtered_metadata = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                # Check if this looks like an episodic trace by checking for episodic-specific fields
                if ('turn_id' in metadata and 'user_text' in metadata and 
                    'mood_family' in metadata and 'raw_vector_pre' in metadata):
                    filtered_metadata.append(metadata)
        
        retrieved_traces = []
        if filtered_metadata:
            # Process the filtered episodic traces
            for metadata in filtered_metadata:
                try:
                    # Check if this is actually an episodic trace by looking for required fields
                    if not all(field in metadata for field in ['turn_id', 'user_text', 'mood_family']):
                        logger.debug(f"Skipping non-episodic entry in session {session_id}")
                        continue
                    
                    # Convert turn_id from string to int if needed
                    if isinstance(metadata.get('turn_id'), str):
                        try:
                            if metadata['turn_id'].startswith('turn_'):
                                metadata['turn_id'] = int(metadata['turn_id'].replace('turn_', ''))
                            else:
                                metadata['turn_id'] = int(metadata['turn_id'])
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid turn_id format: {metadata.get('turn_id')}")
                            continue
                    
                    # Convert JSON strings back to original data structures
                    if 'raw_vector_pre' in metadata and isinstance(metadata['raw_vector_pre'], str):
                        try:
                            metadata['raw_vector_pre'] = json.loads(metadata['raw_vector_pre'])
                        except (json.JSONDecodeError, TypeError):
                            metadata['raw_vector_pre'] = []
                            
                    if 'raw_vector_post' in metadata and isinstance(metadata['raw_vector_post'], str):
                        try:
                            metadata['raw_vector_post'] = json.loads(metadata['raw_vector_post'])
                        except (json.JSONDecodeError, TypeError):
                            metadata['raw_vector_post'] = []
                            
                    if 'dimension_snapshot' in metadata and isinstance(metadata['dimension_snapshot'], str):
                        try:
                            metadata['dimension_snapshot'] = json.loads(metadata['dimension_snapshot'])
                        except (json.JSONDecodeError, TypeError):
                            metadata['dimension_snapshot'] = {}
                            
                    if 'param_modulation' in metadata and isinstance(metadata['param_modulation'], str):
                        try:
                            metadata['param_modulation'] = json.loads(metadata['param_modulation'])
                        except (json.JSONDecodeError, TypeError):
                            metadata['param_modulation'] = {}
                            
                    if 'applied_seeds' in metadata and isinstance(metadata['applied_seeds'], str):
                        try:
                            metadata['applied_seeds'] = json.loads(metadata['applied_seeds'])
                        except (json.JSONDecodeError, TypeError):
                            metadata['applied_seeds'] = []
                    
                    # Set defaults for missing fields to ensure valid EpisodicTrace
                    metadata.setdefault('distortion_type', '')
                    metadata.setdefault('distorted_meaning', '')
                    metadata.setdefault('interpretation_delta', '')
                    metadata.setdefault('intensity', 0.0)
                    metadata.setdefault('core_emotion', None)
                    
                    # Validate we have all required fields before creating EpisodicTrace
                    required_fields = ['turn_id', 'user_text', 'raw_vector_pre', 'raw_vector_post', 
                                     'mood_family', 'distortion_type', 'dimension_snapshot', 
                                     'interpretation_delta', 'applied_seeds', 'param_modulation']
                    
                    if all(field in metadata for field in required_fields):
                        retrieved_traces.append(EpisodicTrace(**metadata))
                    else:
                        missing = [f for f in required_fields if f not in metadata]
                        logger.debug(f"Skipping entry missing fields: {missing}")
                        
                except Exception as trace_error:
                    logger.debug(f"Skipping invalid episodic trace entry: {trace_error}")
                    continue
        
        # Sort by turn_id descending (most recent first) and limit
        retrieved_traces.sort(key=lambda t: t.turn_id, reverse=True)
        retrieved_traces = retrieved_traces[:limit]
        
        logger.debug(f"Retrieved {len(retrieved_traces)} episodic traces for session {session_id}")
        return retrieved_traces
        
    except Exception as e:
        logger.error(f"Failed to retrieve episodic traces for session {session_id}: {e}", exc_info=True)
        return []

async def create_and_store_episodic_trace(
    session_id: str,
    turn_id: int,
    user_input: str,
    ai_response: str,
    user_affect: List[float],
    self_affect: List[float],
    emotional_state: Optional[Any] = None,
    context_synopses: Optional[List[str]] = None,
    reflection: Optional[str] = None
) -> bool:
    """
    Creates and stores an episodic trace for a completed turn.
    
    Args:
        session_id: Session identifier
        turn_id: Turn number
        user_input: User's message
        ai_response: AI's response
        user_affect: User affect vector (28-dim)
        self_affect: AI's affect vector (28-dim) 
        emotional_state: EmotionState object if available
        context_synopses: Context memories used
        reflection: Optional reflection on the turn
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract emotional state information if available
        mood_family = "neutral"
        distortion_type = ""
        intensity = 0.0
        core_emotion = None
        dimension_snapshot = {}
        applied_seeds = []
        param_modulation = {}
        
        if emotional_state:
            mood_family = getattr(emotional_state, 'mood_family', 'neutral')
            intensity = getattr(emotional_state, 'intensity', 0.0)
            core_emotion = getattr(emotional_state, 'dominant_label', None)
            
            # Create dimension snapshot from vector_28 if available
            if hasattr(emotional_state, 'vector_28') and emotional_state.vector_28:
                dimension_snapshot = {f"dim_{i}": val for i, val in enumerate(emotional_state.vector_28)}
        
        # Create episodic trace
        episodic_trace = EpisodicTrace(
            turn_id=turn_id,
            user_text=user_input,
            raw_vector_pre=user_affect,
            raw_vector_post=self_affect,
            mood_family=mood_family,
            distortion_type=distortion_type,
            distorted_meaning=None,
            dimension_snapshot=dimension_snapshot,
            interpretation_delta=reflection or f"Turn {turn_id} in session {session_id}",
            applied_seeds=applied_seeds,
            param_modulation=param_modulation,
            session_id=session_id,
            intensity=intensity,
            core_emotion=core_emotion
        )
        
        # Store the trace
        await save_episodic_trace(episodic_trace)
        
        logger.info(f"✅ Created and stored episodic trace for session {session_id}, turn {turn_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create episodic trace for session {session_id}, turn {turn_id}: {e}")
        return False

async def retrieve_episodic_traces(query: str, k: int, mood_family_filter: Optional[str] = None) -> List[EpisodicTrace]:
    """
    Retrieves the k most relevant episodic traces from ChromaDB using semantic search,
    with an optional filter for mood family.
    """
    if not config.embedder:
        logger.error("Embedder not initialized, cannot retrieve episodic traces.")
        return []

    try:
        collection = get_episodic_collection()
        
        query_embedding = config.embedder.encode(query).tolist()
        
        where_clause = {}
        if mood_family_filter:
            where_clause = {"mood_family": mood_family_filter}
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=["metadatas"]
        )
        
        retrieved_traces = []
        if results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                # Convert JSON strings back to original data structures
                if 'raw_vector_pre' in metadata and isinstance(metadata['raw_vector_pre'], str):
                    try:
                        metadata['raw_vector_pre'] = json.loads(metadata['raw_vector_pre'])
                    except (json.JSONDecodeError, TypeError):
                        metadata['raw_vector_pre'] = None
                        
                if 'raw_vector_post' in metadata and isinstance(metadata['raw_vector_post'], str):
                    try:
                        metadata['raw_vector_post'] = json.loads(metadata['raw_vector_post'])
                    except (json.JSONDecodeError, TypeError):
                        metadata['raw_vector_post'] = None
                        
                if 'dimension_snapshot' in metadata and isinstance(metadata['dimension_snapshot'], str):
                    try:
                        metadata['dimension_snapshot'] = json.loads(metadata['dimension_snapshot'])
                    except (json.JSONDecodeError, TypeError):
                        metadata['dimension_snapshot'] = {}
                        
                if 'param_modulation' in metadata and isinstance(metadata['param_modulation'], str):
                    try:
                        metadata['param_modulation'] = json.loads(metadata['param_modulation'])
                    except (json.JSONDecodeError, TypeError):
                        metadata['param_modulation'] = {}
                        
                if 'applied_seeds' in metadata and isinstance(metadata['applied_seeds'], str):
                    try:
                        metadata['applied_seeds'] = json.loads(metadata['applied_seeds'])
                    except (json.JSONDecodeError, TypeError):
                        metadata['applied_seeds'] = []
                        
                retrieved_traces.append(EpisodicTrace(**metadata))
        
        logger.debug(f"Retrieved {len(retrieved_traces)} traces for query '{query}'")
        return retrieved_traces
        
    except Exception as e:
        logger.error(f"Failed to retrieve episodic traces from ChromaDB: {e}", exc_info=True)
        return [] 