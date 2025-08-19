#!/usr/bin/env python3
"""
Migration script to add judge_profile column to judge_configurations table.
This script adds the new judge_profile column and sets default values for existing records.
"""

import sqlite3
import sys
import os

def migrate_judge_profiles():
    """Add judge_profile column to judge_configurations table"""
    
    # Database path
    db_path = "data/simple_eval.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Creating new database schema will include judge_profile column.")
        return True
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if judge_profile column already exists
        cursor.execute("PRAGMA table_info(judge_configurations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'judge_profile' in columns:
            print("✓ judge_profile column already exists in judge_configurations table")
            return True
        
        print("Adding judge_profile column to judge_configurations table...")
        
        # Add the judge_profile column with default value 'simple'
        cursor.execute('''
            ALTER TABLE judge_configurations 
            ADD COLUMN judge_profile TEXT NOT NULL DEFAULT 'simple'
        ''')
        
        # Update any existing records to have 'simple' profile
        cursor.execute('''
            UPDATE judge_configurations 
            SET judge_profile = 'simple' 
            WHERE judge_profile IS NULL OR judge_profile = ''
        ''')
        
        # Commit changes
        conn.commit()
        
        print("✓ Successfully added judge_profile column")
        print("✓ Set default profile 'simple' for existing judge configurations")
        
        # Verify the migration
        cursor.execute("SELECT COUNT(*) FROM judge_configurations WHERE judge_profile IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} judge configurations have judge_profile set")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def main():
    """Main migration function"""
    print("Judge Profile Migration Script")
    print("=" * 50)
    
    success = migrate_judge_profiles()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("\nNew judge configurations can now use profiles:")
        print("  • simple: Only check if answer is correct")
        print("  • detailed: Check correctness, completeness, no extra facts, answer type match")
        print("  • custom: Use your own prompt")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()