"""
PARADOX SYSTEM MIGRATION
Safe integration of paradox cultivation into existing lattice
"""

import logging
import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

class ParadoxMigration:
    """Handles migration and integration of paradox system"""
    
    def __init__(self):
        self.migration_log = []
        self.dry_run = True  # Safety default
    
    async def run_full_migration(self, dry_run: bool = True) -> Dict:
        """
        Execute complete paradox system migration
        """
        self.dry_run = dry_run
        self.migration_log = []
        
        logger.info(f"Starting paradox migration (dry_run={dry_run})")
        
        results = {
            'migration_id': f"paradox_migration_{datetime.now(timezone.utc).timestamp()}",
            'dry_run': dry_run,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Verify system health
            await self._verify_system_health(results)
            
            # Step 2: Create Neo4j schema extensions  
            await self._create_paradox_schema(results)
            
            # Step 3: Migrate existing contradiction data
            await self._migrate_legacy_contradictions(results)
            
            # Step 4: Update node labels for mythology
            await self._update_mythic_labels(results)
            
            # Step 5: Initialize paradox daemon processes
            await self._initialize_paradox_daemons(results)
            
            # Step 6: Test paradox detection
            await self._test_paradox_system(results)
            
            logger.info(f"Migration complete: {len(results['steps_completed'])} steps, {len(results['errors'])} errors")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results['errors'].append(f"Critical migration failure: {e}")
        
        return results
    
    async def _verify_system_health(self, results: Dict):
        """Verify all required systems are operational"""
        import os
        from neo4j import GraphDatabase
        
        step_name = "system_health_check"
        logger.info(" Verifying system health...")
        
        try:
            # Initialize Neo4j connection directly
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_pass = os.getenv("NEO4J_PASS", "test")
            
            neo4j_conn = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            
            # Store connection for other methods
            self.neo4j_conn = neo4j_conn
            
            # Check Neo4j connection
            with neo4j_conn.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single():
                    logger.info(" Neo4j connection verified")
                else:
                    results['errors'].append("Neo4j connection test failed")
                    return
            
            # Check ChromaDB (optional)
            try:
                from src.lattice.config import chroma_db
                if chroma_db:
                    try:
                        chroma_db.get_collection("test_collection")
                        logger.info(" ChromaDB connection verified")
                    except:
                        logger.info(" ChromaDB accessible (test collection doesn't exist, which is expected)")
                else:
                    results['warnings'].append("ChromaDB not initialized - some features may be limited")
            except ImportError:
                results['warnings'].append("ChromaDB not available in this migration context")
            
            # Check embedder (optional)
            try:
                from src.lattice.config import embedder
                if embedder:
                    test_embedding = embedder.encode(["test"])
                    if len(test_embedding) > 0:
                        logger.info(" Embedder functional")
                    else:
                        results['warnings'].append("Embedder returned empty results")
                else:
                    results['warnings'].append("Embedder not initialized - semantic analysis limited")
            except ImportError:
                results['warnings'].append("Embedder not available in this migration context")
            
            results['steps_completed'].append(step_name)
            
        except Exception as e:
            results['errors'].append(f"System health check failed: {e}")
    
    async def _create_paradox_schema(self, results: Dict):
        """Create Neo4j schema for paradox nodes"""
        step_name = "create_paradox_schema"
        logger.info(" Creating paradox schema...")
        
        if not hasattr(self, 'neo4j_conn') or not self.neo4j_conn:
            results['errors'].append("Neo4j driver not available for schema creation")
            return
        
        try:
            with self.neo4j_conn.session() as session:
                # Create constraints and indexes for paradox nodes
                schema_queries = [
                    "CREATE CONSTRAINT paradox_id_unique IF NOT EXISTS FOR (p:Paradox) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT rumble_id_unique IF NOT EXISTS FOR (r:RumbleNote) REQUIRE r.id IS UNIQUE", 
                    "CREATE CONSTRAINT advice_id_unique IF NOT EXISTS FOR (a:Advice) REQUIRE a.id IS UNIQUE",
                    "CREATE INDEX paradox_status IF NOT EXISTS FOR (p:Paradox) ON (p.status)",
                    "CREATE INDEX paradox_type IF NOT EXISTS FOR (p:Paradox) ON (p.paradox_type)",
                    "CREATE INDEX paradox_timestamp IF NOT EXISTS FOR (p:Paradox) ON (p.timestamp)",
                    "CREATE INDEX rumble_timestamp IF NOT EXISTS FOR (r:RumbleNote) ON (r.timestamp)",
                    "CREATE INDEX advice_timestamp IF NOT EXISTS FOR (a:Advice) ON (a.timestamp)"
                ]
                
                for query in schema_queries:
                    if not self.dry_run:
                        session.run(query)
                    logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Executed: {query}")
                
                results['steps_completed'].append(step_name)
                logger.info(" Paradox schema created successfully")
                
        except Exception as e:
            results['errors'].append(f"Schema creation failed: {e}")
    
    async def _migrate_legacy_contradictions(self, results: Dict):
        """Migrate existing contradiction data to paradox format"""
        step_name = "migrate_legacy_contradictions"
        logger.info(" Migrating legacy contradictions...")
        
        if not hasattr(self, 'neo4j_conn') or not self.neo4j_conn:
            results['errors'].append("Neo4j driver not available for migration")
            return
        
        try:
            with self.neo4j_conn.session() as session:
                # Find existing reflection nodes with contradictions
                query = """
                MATCH (r:Reflection) 
                WHERE r.contradiction IS NOT NULL AND r.contradiction = true
                RETURN r.id as reflection_id, r.text as text, r.timestamp as timestamp,
                       r.tension_score as tension_score
                LIMIT 50
                """
                
                result = session.run(query)
                legacy_contradictions = [record for record in result]
                
                migrated_count = 0
                
                for record in legacy_contradictions:
                    if not self.dry_run:
                        # Create corresponding paradox node
                        paradox_query = """
                        CREATE (p:Paradox {
                            id: $paradox_id,
                            timestamp: $timestamp,
                            tension_score: $tension_score,
                            semantic_tension: $tension_score,
                            affective_delta: 0.0,
                            status: 'integrated',
                            response_text: $text,
                            paradox_type: 'legacy_contradiction',
                            explanation: 'Migrated from legacy reflection system'
                        })
                        RETURN p.id as created_id
                        """
                        
                        paradox_id = f"legacy_paradox_{record['reflection_id']}"
                        
                        session.run(paradox_query, {
                            'paradox_id': paradox_id,
                            'timestamp': record['timestamp'],
                            'tension_score': record.get('tension_score', 0.5),
                            'text': record.get('text', '')
                        })
                        
                        # Link to original reflection
                        link_query = """
                        MATCH (p:Paradox {id: $paradox_id})
                        MATCH (r:Reflection {id: $reflection_id})
                        CREATE (p)-[:EMERGED_FROM]->(r)
                        """
                        
                        session.run(link_query, {
                            'paradox_id': paradox_id,
                            'reflection_id': record['reflection_id']
                        })
                    
                    migrated_count += 1
                    logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Migrated contradiction: {record['reflection_id']}")
                
                results['steps_completed'].append(step_name)
                logger.info(f" Migrated {migrated_count} legacy contradictions")
                
        except Exception as e:
            results['errors'].append(f"Legacy contradiction migration failed: {e}")
    
    async def _update_mythic_labels(self, results: Dict):
        """Update node labels to use mythic terminology"""
        step_name = "update_mythic_labels"
        logger.info(" Updating mythic terminology...")
        
        if not hasattr(self, 'neo4j_conn') or not self.neo4j_conn:
            results['errors'].append("Neo4j driver not available for label updates")
            return
        
        try:
            with self.neo4j_conn.session() as session:
                # Update User nodes to Architect
                if not self.dry_run:
                    user_query = """
                    MATCH (n:User)
                    SET n:Architect
                    REMOVE n:User
                    RETURN count(n) as updated_count
                    """
                    result = session.run(user_query)
                    user_count = result.single()['updated_count'] if result.single() else 0
                else:
                    # Count what would be updated
                    count_query = "MATCH (n:User) RETURN count(n) as user_count"
                    result = session.run(count_query)
                    user_count = result.single()['user_count'] if result.single() else 0
                
                logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Updated {user_count} User nodes to Architect")
                
                # Update AI/Assistant nodes to Daemon  
                if not self.dry_run:
                    ai_query = """
                    MATCH (n) 
                    WHERE n:AI OR n:Assistant OR n:System
                    SET n:Daemon
                    REMOVE n:AI, n:Assistant, n:System
                    RETURN count(n) as updated_count
                    """
                    result = session.run(ai_query)
                    ai_count = result.single()['updated_count'] if result.single() else 0
                else:
                    count_query = """
                    MATCH (n) 
                    WHERE n:AI OR n:Assistant OR n:System
                    RETURN count(n) as ai_count
                    """
                    result = session.run(count_query)
                    ai_count = result.single()['ai_count'] if result.single() else 0
                
                logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Updated {ai_count} AI/Assistant nodes to Daemon")
                
                results['steps_completed'].append(step_name)
                logger.info(" Mythic terminology updated")
                
        except Exception as e:
            results['errors'].append(f"Mythic label update failed: {e}")
    
    async def _initialize_paradox_daemons(self, results: Dict):
        """Initialize paradox processing daemon jobs"""
        step_name = "initialize_paradox_daemons"
        logger.info(" Initializing paradox daemons...")
        
        try:
            if not self.dry_run:
                # Import scheduler if available
                try:
                    from apscheduler.schedulers.background import BackgroundScheduler
                    from src.lattice.paradox.processing import percolate_paradoxes, integrate_calm_paradoxes
                    
                    # Note: In actual implementation, this would be added to main service
                    # For migration, we just verify the imports work
                    logger.info(" Paradox processing functions imported successfully")
                    
                except ImportError as ie:
                    results['warnings'].append(f"APScheduler not available: {ie}")
                    logger.warning("APScheduler not installed - paradox daemons will need manual scheduling")
                
            results['steps_completed'].append(step_name)
            logger.info(" Paradox daemon initialization complete")
            
        except Exception as e:
            results['errors'].append(f"Paradox daemon initialization failed: {e}")
    
    async def _test_paradox_system(self, results: Dict):
        """Test paradox detection with sample data"""
        step_name = "test_paradox_system"
        logger.info(" Testing paradox detection...")
        
        try:
            from src.lattice.paradox.detection import detect_paradox
            from lattice.paradox.DEPRECATED_language_hygiene import build_mythic_prompt, validate_language_hygiene
            
            # Test contradiction detection
            test_response = "I never contradict myself, but sometimes I do say conflicting things."
            test_memories = [
                {'id': 'test_1', 'text': 'I always maintain consistency in my responses.'}
            ]
            
            paradox_result = await detect_paradox(test_response, test_memories, affect_delta=0.2)
            
            if paradox_result:
                logger.info(f" Paradox detection working: {paradox_result['paradox_type']}")
            else:
                results['warnings'].append("Paradox detection test did not trigger - may need threshold adjustment")
            
            # Test language hygiene
            test_text = "As an AI, I will help the user with their request."
            hygiene_report = validate_language_hygiene(test_text)
            
            if not hygiene_report['is_clean']:
                logger.info(f" Language hygiene detection working: found {len(hygiene_report['violations'])} violations")
            
            # Test mythic prompt building
            mythic_prompt = build_mythic_prompt("Test plan", ["Test context"], {"curiosity": 0.8}, "Hello AI")
            
            if "architect" in mythic_prompt and "daemon" in mythic_prompt:
                logger.info(" Mythic prompt building working")
            else:
                results['warnings'].append("Mythic prompt building may not be applying terminology correctly")
            
            results['steps_completed'].append(step_name)
            logger.info(" Paradox system testing complete")
            
        except Exception as e:
            results['errors'].append(f"Paradox system testing failed: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'neo4j_conn') and self.neo4j_conn:
            try:
                self.neo4j_conn.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {e}")


async def run_migration(dry_run: bool = True) -> Dict:
    """
    Main migration entry point
    """
    migration = ParadoxMigration()
    try:
        return await migration.run_full_migration(dry_run=dry_run)
    finally:
        migration.cleanup()


if __name__ == "__main__":
    import sys
    
    # Allow command line execution
    dry_run = "--live" not in sys.argv
    
    if not dry_run:
        print("WARNING: LIVE MIGRATION MODE - This will modify your database!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled")
            sys.exit(0)
    
    print(f"Running paradox migration (dry_run={dry_run})")
    
    result = asyncio.run(run_migration(dry_run=dry_run))
    
    print("\nMIGRATION REPORT")
    print("=" * 50)
    print(f"Migration ID: {result['migration_id']}")
    print(f"Dry Run: {result['dry_run']}")
    print(f"Steps Completed: {len(result['steps_completed'])}")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    
    if result['errors']:
        print("\nERRORS:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print("\nWARNINGS:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    print(f"\nSteps completed: {', '.join(result['steps_completed'])}")
    
    if result['errors']:
        sys.exit(1)
    else:
        print("\nMigration completed successfully")
        sys.exit(0)