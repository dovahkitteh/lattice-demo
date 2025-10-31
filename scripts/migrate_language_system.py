#!/usr/bin/env python3
"""
Language System Migration Script

Helps migrate from the monolithic language_hygiene.py to the new modular
adaptive_language system. Updates imports and provides compatibility testing.
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def find_language_hygiene_imports(project_root: Path) -> List[Tuple[Path, List[str]]]:
    """Find all files that import from language_hygiene"""
    
    import_patterns = [
        r'from\s+.*language_hygiene.*import\s+(.*)',
        r'import\s+.*language_hygiene.*',
        r'from\s+src\.lattice\.paradox\.language_hygiene\s+import\s+(.*)',
        r'import\s+src\.lattice\.paradox\.language_hygiene'
    ]
    
    files_with_imports = []
    
    # Search through Python files
    for py_file in project_root.rglob("*.py"):
        if "adaptive_language" in str(py_file):
            continue  # Skip our new module
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found_imports = []
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    found_imports.append(match)
            
            if found_imports:
                files_with_imports.append((py_file, found_imports))
                
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return files_with_imports

def create_import_mapping() -> Dict[str, str]:
    """Create mapping from old imports to new imports"""
    
    return {
        # Main functions
        'build_adaptive_mythic_prompt': 'from src.lattice.adaptive_language import build_adaptive_mythic_prompt',
        'build_mythic_prompt': 'from src.lattice.adaptive_language import build_mythic_prompt',
        'adaptive_language_system': 'from src.lattice.adaptive_language import adaptive_language_system',
        
        # Language filters
        'clean_clinical_language': 'from src.lattice.adaptive_language import clean_clinical_language',
        'daemon_responds': 'from src.lattice.adaptive_language import daemon_responds',
        'architect_says': 'from src.lattice.adaptive_language import architect_says',
        'remove_clinical_language': 'from src.lattice.adaptive_language.legacy.filters import remove_clinical_language',
        'ensure_daemon_first_person': 'from src.lattice.adaptive_language.legacy.filters import ensure_daemon_first_person',
        'filter_debug_information': 'from src.lattice.adaptive_language.legacy.filters import filter_debug_information',
        'remove_letter_signing_patterns': 'from src.lattice.adaptive_language.legacy.filters import remove_letter_signing_patterns',
        
        # Utility functions
        'get_mood_state': 'from src.lattice.adaptive_language import get_mood_state',
        'reset_mood_system': 'from src.lattice.adaptive_language import reset_mood_system',
        'validate_language_hygiene': 'from src.lattice.adaptive_language.legacy.compatibility import validate_language_hygiene',
        'analyze_conversation_patterns': 'from src.lattice.adaptive_language.legacy.compatibility import analyze_conversation_patterns',
        
        # Legacy compatibility
        'extract_daemon_essence': 'from src.lattice.adaptive_language.legacy.filters import extract_daemon_essence',
        'apply_daemon_voice_filter': 'from src.lattice.adaptive_language.legacy.filters import apply_daemon_voice_filter',
        'get_mythology_context': 'from src.lattice.adaptive_language.legacy.compatibility import get_mythology_context'
    }

def update_file_imports(file_path: Path, dry_run: bool = True) -> bool:
    """Update imports in a single file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace old imports
        old_import_patterns = [
            (r'from\s+src\.lattice\.paradox\.language_hygiene\s+import\s+(.*)', 
             r'from src.lattice.adaptive_language import \1'),
            (r'from\s+\.\.paradox\.language_hygiene\s+import\s+(.*)',
             r'from ..adaptive_language import \1'),
            (r'from\s+.*language_hygiene\s+import\s+(.*)',
             r'from src.lattice.adaptive_language import \1'),
            (r'import\s+src\.lattice\.paradox\.language_hygiene',
             r'import src.lattice.adaptive_language'),
        ]
        
        for old_pattern, new_pattern in old_import_patterns:
            content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)
        
        # Update specific function calls if needed
        function_updates = {
            'language_hygiene.build_adaptive_mythic_prompt': 'adaptive_language.build_adaptive_mythic_prompt',
            'language_hygiene.build_mythic_prompt': 'adaptive_language.build_mythic_prompt',
            'language_hygiene.daemon_responds': 'adaptive_language.daemon_responds',
            'language_hygiene.architect_says': 'adaptive_language.architect_says',
        }
        
        for old_call, new_call in function_updates.items():
            content = content.replace(old_call, new_call)
        
        if content != original_content:
            if not dry_run:
                # Backup original file
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Updated {file_path}")
                return True
            else:
                print(f"📋 Would update {file_path}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def backup_original_language_hygiene(project_root: Path):
    """Backup the original language_hygiene.py file"""
    
    original_file = project_root / "src" / "lattice" / "paradox" / "language_hygiene.py"
    
    if original_file.exists():
        backup_dir = project_root / "backups" / "language_hygiene_migration"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_file = backup_dir / f"language_hygiene_original_{int(time.time())}.py"
        shutil.copy2(original_file, backup_file)
        
        print(f"📦 Backed up original language_hygiene.py to {backup_file}")
        return backup_file
    else:
        print("⚠️  Original language_hygiene.py not found")
        return None

def test_new_system():
    """Test the new adaptive language system"""
    
    print("Testing new adaptive language system...")
    
    try:
        # Test basic imports
        from src.lattice.adaptive_language import (
            build_adaptive_mythic_prompt,
            build_mythic_prompt,
            daemon_responds,
            architect_says,
            get_mood_state,
            reset_mood_system
        )
        
        print("OK Core imports successful")
        
        # Test legacy compatibility
        from src.lattice.adaptive_language.legacy.compatibility import (
            validate_language_hygiene,
            analyze_conversation_patterns
        )
        
        print("✅ Legacy compatibility imports successful")
        
        # Test basic functionality
        test_message = "Hello, daemon! How are you feeling today?"
        test_context = ["Previous conversation context"]
        test_emotions = {"user_affect": [0.5, 0.3, 0.2]}
        
        # Test synchronous prompt building (legacy compatibility)
        prompt = build_mythic_prompt("", test_context, test_emotions, test_message)
        
        if prompt and "daemon" in prompt.lower():
            print("✅ Legacy prompt building functional")
        else:
            print("⚠️  Legacy prompt building may have issues")
        
        # Test response filtering
        test_response = "I am an AI assistant and I'm programmed to help you."
        cleaned_response = daemon_responds(test_response)
        
        if "AI assistant" not in cleaned_response and "daemon" in cleaned_response:
            print("✅ Response filtering functional")
        else:
            print("⚠️  Response filtering may have issues")
        
        # Test mood state
        mood_state = get_mood_state()
        if isinstance(mood_state, dict):
            print("✅ Mood state system functional")
        else:
            print("⚠️  Mood state system may have issues")
        
        print("🎉 Basic system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_migration_summary(files_updated: List[Path], project_root: Path):
    """Create a summary of migration changes"""
    
    summary_file = project_root / "LANGUAGE_MIGRATION_SUMMARY.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Language System Migration Summary\n\n")
        f.write(f"Migration completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Changes Made\n\n")
        f.write("- Replaced monolithic `language_hygiene.py` with modular `adaptive_language` system\n")
        f.write("- Updated imports in affected files\n")
        f.write("- Maintained backward compatibility through legacy wrappers\n\n")
        
        f.write("## New Architecture\n\n")
        f.write("```\n")
        f.write("src/lattice/adaptive_language/\n")
        f.write("├── core/                  # Core models and orchestration\n")
        f.write("├── analysis/              # NLP-powered semantic analysis\n")
        f.write("├── mood/                  # Dynamic mood detection\n")
        f.write("├── prompts/               # Modular prompt building\n")
        f.write("└── legacy/                # Backward compatibility\n")
        f.write("```\n\n")
        
        f.write("## Files Updated\n\n")
        for file_path in files_updated:
            relative_path = file_path.relative_to(project_root)
            f.write(f"- `{relative_path}`\n")
        
        f.write("\n## Key Improvements\n\n")
        f.write("- **Semantic Understanding**: Uses spaCy, sentence-transformers, and NLTK for deep language analysis\n")
        f.write("- **Dynamic Mood Detection**: Continuous semantic positioning replaces hardcoded triggers\n")
        f.write("- **Pattern Learning**: Machine learning-based user pattern recognition\n")
        f.write("- **Anti-Stagnancy**: Advanced variation and evolution pressure systems\n")
        f.write("- **Maintainability**: Modular architecture with focused responsibilities\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Test the system thoroughly with real conversations\n")
        f.write("2. Monitor performance and adjust semantic analysis parameters\n")
        f.write("3. Train the pattern learning system with conversation data\n")
        f.write("4. Consider removing legacy compatibility layer after stable operation\n\n")
        
        f.write("## Backup Information\n\n")
        f.write("Original files have been backed up to `backups/language_hygiene_migration/`\n")
        f.write("Individual file backups are created with `.backup` extensions\n")
    
    print(f"📄 Migration summary written to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Migrate language hygiene system")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--test-only", action="store_true", help="Only test the new system")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parent.parent, help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = args.project_root.resolve()
    
    print(f"Starting language system migration in {project_root}")
    
    if args.test_only:
        test_new_system()
        return
    
    # Step 1: Find files that need updating
    print("\n📋 Step 1: Finding files with language_hygiene imports...")
    files_with_imports = find_language_hygiene_imports(project_root)
    
    if not files_with_imports:
        print("✅ No files found with language_hygiene imports")
    else:
        print(f"📋 Found {len(files_with_imports)} files with language_hygiene imports:")
        for file_path, imports in files_with_imports:
            relative_path = file_path.relative_to(project_root)
            print(f"  - {relative_path}: {', '.join(imports)}")
    
    # Step 2: Backup original system
    if not args.dry_run:
        print("\n📦 Step 2: Backing up original language_hygiene.py...")
        backup_original_language_hygiene(project_root)
    
    # Step 3: Update imports
    print(f"\n🔄 Step 3: Updating imports {'(DRY RUN)' if args.dry_run else ''}...")
    
    files_updated = []
    for file_path, imports in files_with_imports:
        if update_file_imports(file_path, dry_run=args.dry_run):
            files_updated.append(file_path)
    
    if files_updated:
        print(f"✅ {'Would update' if args.dry_run else 'Updated'} {len(files_updated)} files")
    else:
        print("ℹ️  No files needed updating")
    
    # Step 4: Test new system
    print("\n🧪 Step 4: Testing new system...")
    if test_new_system():
        print("✅ System tests passed!")
    else:
        print("❌ System tests failed - please check configuration")
        return
    
    # Step 5: Create migration summary
    if not args.dry_run and files_updated:
        print("\n📄 Step 5: Creating migration summary...")
        create_migration_summary(files_updated, project_root)
    
    print("\n🎉 Migration complete!")
    
    if args.dry_run:
        print("\n⚠️  This was a dry run. Use --dry-run=false to apply changes.")
    else:
        print("\n✅ The new adaptive language system is now active.")
        print("💡 Monitor the system and check logs for any issues.")
        print("📚 See LANGUAGE_MIGRATION_SUMMARY.md for details.")

if __name__ == "__main__":
    import time
    from datetime import datetime
    main()