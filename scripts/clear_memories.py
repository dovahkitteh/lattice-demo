#!/usr/bin/env python3
"""
Enhanced Memory Clearing Script for Lucifer Lattice
Finds and clears ALL memory data from ChromaDB and Neo4j databases.
"""

import os
import shutil
import json
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def find_chromadb_locations():
    """Find all possible ChromaDB storage locations"""
    locations = []
    
    # Common ChromaDB locations
    possible_paths = [
        "./data/",
        "./chroma_db/", 
        "./data/lattice/",
        "./lattice_data/",
        os.path.expanduser("~/.chroma/"),
        os.path.expanduser("~/chroma_db/"),
        "./",  # Current directory for any .chroma files
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            locations.append(abs_path)
            print(f"📁 Found potential ChromaDB location: {abs_path}")
    
    # Look for any chroma-related directories in current and parent directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if "chroma" in dir_name.lower() or "lattice" in dir_name.lower():
                full_path = os.path.join(root, dir_name)
                abs_path = os.path.abspath(full_path)
                if abs_path not in locations:
                    locations.append(abs_path)
                    print(f"📁 Found ChromaDB-related directory: {abs_path}")
    
    return locations

def clear_chromadb_programmatically():
    """Clear ChromaDB using the API"""
    try:
        import chromadb
        
        # Try different client configurations
        clients_to_try = [
            lambda: chromadb.PersistentClient(path="./data/lattice"),
            lambda: chromadb.PersistentClient(path="./data"),
            lambda: chromadb.PersistentClient(path="./chroma_db"),
            lambda: chromadb.PersistentClient(),  # Default location
        ]
        
        for i, client_func in enumerate(clients_to_try):
            try:
                client = client_func()
                collections = client.list_collections()
                print(f"🔍 Client {i+1}: Found {len(collections)} collections")
                
                for collection in collections:
                    print(f"  📊 Collection: {collection.name}")
                    count = collection.count()
                    print(f"      Items: {count}")
                    
                    if count > 0:
                        # Delete all items
                        all_items = collection.get()
                        if all_items['ids']:
                            collection.delete(ids=all_items['ids'])
                            print(f"      ✅ Deleted {len(all_items['ids'])} items")
                        
                        # Verify deletion
                        new_count = collection.count()
                        print(f"      📊 New count: {new_count}")
                
            except Exception as e:
                print(f"❌ Client {i+1} failed: {e}")
                continue
                
    except ImportError:
        print("❌ ChromaDB not available for programmatic clearing")

def clear_neo4j():
    """Clear Neo4j database"""
    try:
        from neo4j import GraphDatabase
        
        # Get Neo4j credentials from environment variables
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASS", "")
        
        # Try to connect to Neo4j
        driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        
        with driver.session() as session:
            # Count existing data
            result = session.run("MATCH (n:Memory) RETURN count(n) as count")
            memory_count = result.single()["count"]
            
            result = session.run("MATCH ()-[r:ECHO]->() RETURN count(r) as count")
            echo_count = result.single()["count"]
            
            print(f"🔍 Found {memory_count} Memory nodes and {echo_count} ECHO relationships in Neo4j")
            
            if memory_count > 0 or echo_count > 0:
                # Clear all Memory nodes and ECHO relationships
                session.run("MATCH (n:Memory) DETACH DELETE n")
                print(f"✅ Deleted all Memory nodes and relationships from Neo4j")
                
                # Verify deletion
                result = session.run("MATCH (n:Memory) RETURN count(n) as count")
                new_count = result.single()["count"]
                print(f"📊 Remaining Memory nodes: {new_count}")
            
        driver.close()
        
    except Exception as e:
        print(f"❌ Neo4j clearing failed: {e}")
        print("💡 Make sure Neo4j Desktop is running with the Lucifer database")

def main():
    print("🧠 Lucifer Lattice - Enhanced Memory Clearing Tool")
    print("=" * 60)
    print()
    
    print("⚠️  WARNING: This will permanently delete ALL memory data!")
    print("   - ChromaDB embeddings and metadata")
    print("   - Neo4j Memory nodes and ECHO relationships")
    print("   - Cannot be undone!")
    print()
    
    response = input("Type 'YES' to confirm deletion: ")
    if response != "YES":
        print("❌ Deletion cancelled")
        return

    # --- Step 1: Forcefully delete the primary data directory ---
    print("\n🗑️  STEP 1: Clearing primary ChromaDB directory...")
    primary_db_path = os.path.abspath("./data/lattice")
    if os.path.exists(primary_db_path):
        try:
            shutil.rmtree(primary_db_path)
            print(f"✅ Successfully deleted primary database directory: {primary_db_path}")
        except Exception as e:
            print(f"❌ FAILED to delete primary database directory: {primary_db_path}")
            print(f"   Reason: {e}")
            print("   Please ensure no services are running and try again.")
    else:
        print("✅ Primary database directory does not exist, nothing to clear.")

    # --- Step 2: Clear Neo4j ---
    print(f"\n🗑️  STEP 2: Clearing Neo4j database...")
    clear_neo4j()

    # --- Step 3: Deep scan for any other rogue databases (optional) ---
    print("\n\n🔍 STEP 3: Deep scanning for other ChromaDB instances...")
    locations = find_chromadb_locations()
    
    print(f"\n🗑️  Found {len(locations)} other potential locations to check...")
    
    # First try programmatic clearing for other instances
    clear_chromadb_programmatically()
    
    # Then try filesystem clearing for other instances
    cleared_dirs = 0
    for location in locations:
        # Skip the primary path we already deleted
        if os.path.abspath(location) == primary_db_path:
            continue
        try:
            if os.path.isdir(location):
                # Look for chroma-related files
                chroma_files = []
                for item in os.listdir(location):
                    item_path = os.path.join(location, item)
                    if (item.startswith('chroma') or 
                        item.endswith('.sqlite') or 
                        item.endswith('.sqlite3') or
                        'chroma' in item.lower() or
                        item == 'memories'):
                        chroma_files.append(item_path)
                
                if chroma_files:
                    print(f"🗑️  Clearing {len(chroma_files)} items from {location}:")
                    for file_path in chroma_files:
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                print(f"    ✅ Removed directory: {os.path.basename(file_path)}")
                            else:
                                os.remove(file_path)
                                print(f"    ✅ Removed file: {os.path.basename(file_path)}")
                            cleared_dirs += 1
                        except Exception as e:
                            print(f"    ❌ Failed to remove {file_path}: {e}")
                            
        except Exception as e:
            print(f"❌ Error processing {location}: {e}")
    
    
    print(f"\n📊 STEP 4: Verification...")
    
    # Verify ChromaDB clearing
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./data/lattice")
        try:
            collection = client.get_collection("memories")
            count = collection.count()
            print(f"📊 ChromaDB 'memories' collection: {count} items remaining")
        except:
            print("📊 ChromaDB 'memories' collection: Not found (likely cleared)")
    except:
        print("📊 ChromaDB verification: Unable to check")
    
    # Verify Neo4j clearing
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASS", "")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            result = session.run("MATCH (n:Memory) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"📊 Neo4j Memory nodes: {count} remaining")
        driver.close()
    except:
        print("📊 Neo4j verification: Unable to check")
    
    print("\n✅ Memory clearing process completed!")
    print("\n💡 Next steps:")
    print("   1. Restart the lattice service if it's running")
    print("   2. Check the enhanced emotion tracker for updated counts")
    print("   3. Start fresh conversations to build new memories")

if __name__ == "__main__":
    main() 