#!/usr/bin/env python3
"""
Reset Daemon Personality to Direct, Intelligent Style
This script clears existing personality evolution and applies the new direct, intelligent values.
"""

import os
import json
import shutil
from datetime import datetime

def reset_personality_state():
    """Reset personality state to apply new direct style values"""
    
    # Path to personality data
    personality_files = [
        "./data/daemon/lucifer_personality.json",
        "./data/daemon/",
    ]
    
    print("🔄 Resetting daemon personality to direct, intelligent style...")
    
    # Remove existing personality files to force reinitialization
    for file_path in personality_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"   ✅ Removed {file_path}")
            elif os.path.isdir(file_path):
                # Backup and clear personality files
                backup_dir = f"./data/daemon_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if os.path.exists(file_path):
                    shutil.move(file_path, backup_dir)
                    print(f"   📁 Backed up daemon data to {backup_dir}")
    
    # Create the daemon data directory
    os.makedirs("./data/daemon", exist_ok=True)
    
    print("✅ Personality reset complete!")
    print("\nNEXT STEPS:")
    print("1. Restart your Lattice service")
    print("2. Test with: 'Hello, my love. How are you feeling now?'")
    print("3. You should see direct, intelligent responses instead of purple prose")
    print("\nThe new personality values are:")
    print("- Poetic Expression: 0.3 (HEAVILY REDUCED)")
    print("- Intensity Level: 0.6 (REDUCED)")
    print("- Philosophical Depth: 0.5 (REDUCED)")
    print("- Emotional Depth: 0.6 (REDUCED)")
    print("- Strategic emphasis only when genuinely needed")

if __name__ == "__main__":
    reset_personality_state() 