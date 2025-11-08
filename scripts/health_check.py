#!/usr/bin/env python3
import requests
import json

def check_memory_status():
    """Check detailed memory status"""
    try:
        response = requests.get("http://127.0.0.1:11434/v1/memories/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            print(f"🗃️  Total memories: {data.get('total_memories', 0)}")
            print(f"📊 Database status:")
            print(f"   ChromaDB: {'✅' if data.get('database_components', {}).get('chroma_db') else '❌'}")
            print(f"   Neo4j: {'✅' if data.get('database_components', {}).get('neo4j_conn') else '❌'}")
            
            samples = data.get('sample_memories', {})
            if samples.get('metadatas'):
                print("\n📋 Sample Memory Details:")
                for i, meta in enumerate(samples['metadatas']):
                    origin = meta.get('origin', 'unknown')
                    print(f"\n   Memory {i+1} - Origin: {origin}")
                    print(f"     Has user_affect: {'✅' if meta.get('has_user_affect') else '❌'}")
                    print(f"     Has self_affect: {'✅' if meta.get('has_self_affect') else '❌'}")
                    print(f"     Has user_reflection: {'✅' if meta.get('has_user_reflection') else '❌'}")
                    print(f"     Has self_reflection: {'✅' if meta.get('has_self_reflection') else '❌'}")
                    print(f"     Has legacy_reflection: {'✅' if meta.get('has_legacy_reflection') else '❌'}")
                
                # Count different memory types
                conversation_turns = [m for m in samples['metadatas'] if m.get('origin') == 'conversation_turn']
                external_users = [m for m in samples['metadatas'] if m.get('origin') == 'external_user']
                
                print(f"\n📈 Memory Type Breakdown (in sample):")
                print(f"   conversation_turn: {len(conversation_turns)}")
                print(f"   external_user: {len(external_users)}")
                
                # Check if recent fixes are working
                if conversation_turns:
                    recent = conversation_turns[0]
                    dual_affect_working = recent.get('has_user_affect') and recent.get('has_self_affect')
                    dual_reflection_working = recent.get('has_user_reflection') and recent.get('has_self_reflection')
                    
                    print(f"\n🎯 Fix Status:")
                    print(f"   Dual-affect storage: {'✅ WORKING' if dual_affect_working else '❌ NOT WORKING'}")
                    print(f"   Dual-reflection storage: {'✅ WORKING' if dual_reflection_working else '❌ NOT WORKING'}")
                    
                    if dual_affect_working and dual_reflection_working:
                        print("\n🎉 MEMORY SYSTEM IS FULLY FUNCTIONAL!")
                    elif dual_affect_working:
                        print("\n⚠️  Dual-affect is working, but reflections need attention")
                    else:
                        print("\n🔧 Memory system still needs fixes")
                else:
                    print("\n⚠️  No recent conversation_turn memories found in sample")
                    print("     This might mean the fixes aren't being applied or")
                    print("     the sample only shows old memories")
            
            # Show sample documents too
            if samples.get('documents'):
                print(f"\n📝 Sample Documents:")
                for i, doc in enumerate(samples['documents'][:3]):
                    print(f"   {i+1}. {doc}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_memory_status() 