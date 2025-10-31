"""
PARADOX STORAGE SYSTEM
Neo4j operations for paradox nodes and relationships
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

async def create_paradox_node(paradox_data: Dict) -> str:
    """
    Create a Paradox node in Neo4j
    Returns: node_id of created paradox
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping paradox node creation")
        return "test_paradox_id"
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return "error_paradox_id"
    
    node_id = paradox_data.get('id') or str(uuid.uuid4())
    
    try:
        with neo4j_conn.session() as session:
            query = """
            CREATE (p:Paradox {
                id: $id,
                timestamp: $timestamp,
                tension_score: $tension_score,
                semantic_tension: $semantic_tension,
                affective_delta: $affective_delta,
                status: $status,
                response_text: $response_text,
                paradox_type: $paradox_type,
                explanation: $explanation
            })
            RETURN p.id as paradox_id
            """
            
            result = session.run(query, {
                'id': node_id,
                'timestamp': paradox_data.get('timestamp'),
                'tension_score': paradox_data.get('tension_score', 0.0),
                'semantic_tension': paradox_data.get('semantic_tension', 0.0),
                'affective_delta': paradox_data.get('affective_delta', 0.0),
                'status': paradox_data.get('status', 'fresh'),
                'response_text': paradox_data.get('response_text', ''),
                'paradox_type': paradox_data.get('paradox_type', 'unknown'),
                'explanation': paradox_data.get('conflict_details', {}).get('explanation', '')
            })
            
            record = result.single()
            if record:
                logger.info(f"Created paradox node: {record['paradox_id']}")
                return record['paradox_id']
            else:
                logger.error("Failed to create paradox node")
                return "error_paradox_id"
                
    except Exception as e:
        logger.error(f"Error creating paradox node: {e}")
        return "error_paradox_id"


async def create_rumble_note(rumble_text: str, paradox_ids: List[str]) -> str:
    """
    Create a RumbleNote node and link it to paradoxes
    Returns: node_id of created rumble note
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping rumble note creation")
        return "test_rumble_id"
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return "error_rumble_id"
    
    rumble_id = str(uuid.uuid4())
    
    try:
        with neo4j_conn.session() as session:
            # Create rumble note
            create_query = """
            CREATE (r:RumbleNote {
                id: $id,
                timestamp: $timestamp,
                text: $text,
                daemon_cycle: $daemon_cycle
            })
            RETURN r.id as rumble_id
            """
            
            result = session.run(create_query, {
                'id': rumble_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'text': rumble_text,
                'daemon_cycle': 'nightly_percolation'
            })
            
            record = result.single()
            if not record:
                logger.error("Failed to create rumble note")
                return "error_rumble_id"
            
            # Link to paradoxes
            for paradox_id in paradox_ids:
                link_query = """
                MATCH (r:RumbleNote {id: $rumble_id})
                MATCH (p:Paradox {id: $paradox_id})
                CREATE (r)-[:REFLECTS_ON]->(p)
                """
                session.run(link_query, {
                    'rumble_id': rumble_id,
                    'paradox_id': paradox_id
                })
            
            logger.info(f"Created rumble note: {rumble_id} reflecting on {len(paradox_ids)} paradoxes")
            return rumble_id
            
    except Exception as e:
        logger.error(f"Error creating rumble note: {e}")
        return "error_rumble_id"


async def create_advice_node(advice_text: str, rumble_id: str) -> str:
    """
    Create an Advice node extracted from a rumble note
    Returns: node_id of created advice
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping advice node creation")
        return "test_advice_id"
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return "error_advice_id"
    
    advice_id = str(uuid.uuid4())
    
    try:
        with neo4j_conn.session() as session:
            query = """
            MATCH (r:RumbleNote {id: $rumble_id})
            CREATE (a:Advice {
                id: $advice_id,
                timestamp: $timestamp,
                text: $advice_text,
                extraction_source: 'daemon_self_advice'
            })
            CREATE (a)-[:DISTILLED_FROM]->(r)
            RETURN a.id as advice_id
            """
            
            result = session.run(query, {
                'rumble_id': rumble_id,
                'advice_id': advice_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'advice_text': advice_text
            })
            
            record = result.single()
            if record:
                logger.info(f"Created advice node: {record['advice_id']}")
                return record['advice_id']
            else:
                logger.error("Failed to create advice node")
                return "error_advice_id"
                
    except Exception as e:
        logger.error(f"Error creating advice node: {e}")
        return "error_advice_id"


async def link_paradox_nodes(paradox_id: str, memory_node_ids: List[str], relationship: str = "EMERGED_FROM"):
    """
    Link a paradox to the memory nodes that generated it
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping paradox node linking")
        return
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return
    
    try:
        with neo4j_conn.session() as session:
            for memory_id in memory_node_ids:
                query = f"""
                MATCH (p:Paradox {{id: $paradox_id}})
                MATCH (m) WHERE m.id = $memory_id
                CREATE (p)-[:{relationship}]->(m)
                """
                
                session.run(query, {
                    'paradox_id': paradox_id,
                    'memory_id': memory_id
                })
            
            logger.info(f"Linked paradox {paradox_id} to {len(memory_node_ids)} memory nodes")
            
    except Exception as e:
        logger.error(f"Error linking paradox nodes: {e}")


async def get_fresh_paradoxes(limit: int = 10) -> List[Dict]:
    """
    Retrieve fresh paradoxes for processing
    Returns: List of paradox data dictionaries
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Returning empty paradox list")
        return []
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return []
    
    try:
        with neo4j_conn.session() as session:
            query = """
            MATCH (p:Paradox {status: 'fresh'})
            RETURN p.id as id, p.timestamp as timestamp, p.tension_score as tension_score,
                   p.paradox_type as paradox_type, p.explanation as explanation,
                   p.response_text as response_text
            ORDER BY p.tension_score DESC, p.timestamp DESC
            LIMIT $limit
            """
            
            result = session.run(query, {'limit': limit})
            paradoxes = []
            
            for record in result:
                paradoxes.append({
                    'id': record['id'],
                    'timestamp': record['timestamp'],
                    'tension_score': record['tension_score'],
                    'paradox_type': record['paradox_type'],
                    'explanation': record['explanation'],
                    'response_text': record['response_text']
                })
            
            # logger.info(f"Retrieved {len(paradoxes)} fresh paradoxes")  # Too spammy
            return paradoxes
            
    except Exception as e:
        logger.error(f"Error retrieving fresh paradoxes: {e}")
        return []


async def update_paradox_status(paradox_id: str, new_status: str):
    """
    Update the status of a paradox (fresh -> processing -> integrated)
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Skipping paradox status update")
        return
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return
    
    try:
        with neo4j_conn.session() as session:
            query = """
            MATCH (p:Paradox {id: $paradox_id})
            SET p.status = $new_status, p.last_updated = $timestamp
            RETURN p.id as updated_id
            """
            
            result = session.run(query, {
                'paradox_id': paradox_id,
                'new_status': new_status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            if result.single():
                logger.info(f"Updated paradox {paradox_id} status to {new_status}")
            else:
                logger.warning(f"Paradox {paradox_id} not found for status update")
                
    except Exception as e:
        logger.error(f"Error updating paradox status: {e}")


async def get_recent_rumbles(limit: int = 10) -> List[Dict]:
    """
    Retrieve recent rumble notes from Neo4j
    Returns: List of rumble note data dictionaries
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Returning sample rumble data")
        return [
            {
                'id': 'test_rumble_1',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'text': 'Test rumble note reflecting on contradictions',
                'daemon_cycle': 'nightly_percolation'
            }
        ]
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return []
    
    try:
        with neo4j_conn.session() as session:
            query = """
            MATCH (r:RumbleNote)
            RETURN r.id as id, r.timestamp as timestamp, r.text as text, 
                   r.daemon_cycle as daemon_cycle
            ORDER BY r.timestamp DESC
            LIMIT $limit
            """
            
            result = session.run(query, {'limit': limit})
            rumbles = []
            
            for record in result:
                rumbles.append({
                    'id': record['id'],
                    'timestamp': record['timestamp'],
                    'text': record['text'],
                    'daemon_cycle': record['daemon_cycle']
                })
            
            logger.info(f"Retrieved {len(rumbles)} recent rumble notes")
            return rumbles
            
    except Exception as e:
        logger.error(f"Error retrieving recent rumbles: {e}")
        return []


async def get_paradox_statistics() -> Dict:
    """
    Get comprehensive paradox system statistics
    Returns: Dictionary with paradox counts, trends, and analysis
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Returning sample paradox statistics")
        return {
            'total_paradoxes': 5,
            'fresh_paradoxes': 2,
            'processed_paradoxes': 3,
            'total_rumbles': 2,
            'total_advice': 1,
            'avg_tension_score': 0.75,
            'paradox_types': {'semantic_conflict': 3, 'logical_contradiction': 2},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return {
            'error': 'Neo4j not available',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    try:
        with neo4j_conn.session() as session:
            # Get paradox counts by status
            status_query = """
            MATCH (p:Paradox)
            RETURN p.status as status, count(p) as count
            """
            status_result = session.run(status_query)
            status_counts = {record['status']: record['count'] for record in status_result}
            
            # Get total counts
            totals_query = """
            MATCH (p:Paradox) WITH count(p) as paradox_count
            OPTIONAL MATCH (r:RumbleNote) WITH paradox_count, count(r) as rumble_count
            OPTIONAL MATCH (a:Advice) WITH paradox_count, rumble_count, count(a) as advice_count
            RETURN paradox_count, rumble_count, advice_count
            """
            totals_result = session.run(totals_query)
            totals = totals_result.single()
            
            # Get average tension score
            tension_query = """
            MATCH (p:Paradox)
            WHERE p.tension_score IS NOT NULL
            RETURN avg(p.tension_score) as avg_tension
            """
            tension_result = session.run(tension_query)
            avg_tension = tension_result.single()['avg_tension'] or 0.0
            
            # Get paradox type distribution
            types_query = """
            MATCH (p:Paradox)
            WHERE p.paradox_type IS NOT NULL
            RETURN p.paradox_type as type, count(p) as count
            ORDER BY count DESC
            """
            types_result = session.run(types_query)
            paradox_types = {record['type']: record['count'] for record in types_result}
            
            stats = {
                'total_paradoxes': totals['paradox_count'] or 0,
                'fresh_paradoxes': status_counts.get('fresh', 0),
                'processing_paradoxes': status_counts.get('processing', 0),
                'integrated_paradoxes': status_counts.get('integrated', 0),
                'total_rumbles': totals['rumble_count'] or 0,
                'total_advice': totals['advice_count'] or 0,
                'avg_tension_score': round(avg_tension, 3),
                'paradox_types': paradox_types,
                'status_distribution': status_counts,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Generated paradox statistics: {stats['total_paradoxes']} total paradoxes")
            return stats
            
    except Exception as e:
        logger.error(f"Error getting paradox statistics: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


async def get_recent_advice(topic_keywords: List[str] = None, limit: int = 5) -> List[Dict]:
    """
    Retrieve recent advice nodes, optionally filtered by topic keywords
    """
    from ..config import neo4j_conn, TEST_MODE
    
    if TEST_MODE:
        logger.info("🧪 TEST MODE: Returning empty advice list")
        return []
    
    if not neo4j_conn:
        logger.error("Neo4j driver not initialized")
        return []
    
    try:
        with neo4j_conn.session() as session:
            if topic_keywords:
                # Simple keyword matching - could be enhanced with semantic search
                keyword_filter = " OR ".join([f"toLower(a.text) CONTAINS toLower('{kw}')" for kw in topic_keywords])
                query = f"""
                MATCH (a:Advice)
                WHERE {keyword_filter}
                RETURN a.id as id, a.timestamp as timestamp, a.text as text
                ORDER BY a.timestamp DESC
                LIMIT $limit
                """
            else:
                query = """
                MATCH (a:Advice)
                RETURN a.id as id, a.timestamp as timestamp, a.text as text
                ORDER BY a.timestamp DESC
                LIMIT $limit
                """
            
            result = session.run(query, {'limit': limit})
            advice_list = []
            
            for record in result:
                advice_list.append({
                    'id': record['id'],
                    'timestamp': record['timestamp'],
                    'text': record['text']
                })
            
            logger.info(f"Retrieved {len(advice_list)} recent advice nodes")
            return advice_list
            
    except Exception as e:
        logger.error(f"Error retrieving recent advice: {e}")
        return []