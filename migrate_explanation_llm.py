#!/usr/bin/env python3
"""
Migration script to add is_explanation_llm column to llm_configurations table.
This enables marking LLM configurations for explanation generation use.
"""

import sqlite3
import sys
import os

def migrate_explanation_llm():
    """Add is_explanation_llm column to llm_configurations table"""
    
    # Database path
    db_path = "data/simple_eval.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Creating new database schema will include is_explanation_llm column.")
        return True
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if is_explanation_llm column already exists
        cursor.execute("PRAGMA table_info(llm_configurations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_explanation_llm' in columns:
            print("✓ is_explanation_llm column already exists in llm_configurations table")
            return True
        
        print("Adding is_explanation_llm column to llm_configurations table...")
        
        # Add the is_explanation_llm column with default value False
        cursor.execute('''
            ALTER TABLE llm_configurations 
            ADD COLUMN is_explanation_llm BOOLEAN NOT NULL DEFAULT 0
        ''')
        
        # Update any existing records to have is_explanation_llm = False
        cursor.execute('''
            UPDATE llm_configurations 
            SET is_explanation_llm = 0 
            WHERE is_explanation_llm IS NULL
        ''')
        
        # Commit changes
        conn.commit()
        
        print("✓ Successfully added is_explanation_llm column")
        print("✓ Set all existing LLM configurations as non-explanation LLMs")
        
        # Verify the migration
        cursor.execute("SELECT COUNT(*) FROM llm_configurations WHERE is_explanation_llm = 0")
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} LLM configurations are not explanation LLMs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def main():
    """Main migration function"""
    print("Explanation LLM Field Migration Script")
    print("=" * 50)
    
    success = migrate_explanation_llm()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("\nLLM configurations can now be marked for explanation generation use.")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()