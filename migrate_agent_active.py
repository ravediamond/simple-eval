#!/usr/bin/env python3
"""
Migration script to add is_active column to agents table.
This enables soft deletion of agents.
"""

import sqlite3
import sys
import os

def migrate_agent_active():
    """Add is_active column to agents table"""
    
    # Database path
    db_path = "data/simple_eval.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Creating new database schema will include is_active column.")
        return True
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if is_active column already exists
        cursor.execute("PRAGMA table_info(agents)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_active' in columns:
            print("✓ is_active column already exists in agents table")
            return True
        
        print("Adding is_active column to agents table...")
        
        # Add the is_active column with default value True
        cursor.execute('''
            ALTER TABLE agents 
            ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1
        ''')
        
        # Update any existing records to have is_active = True
        cursor.execute('''
            UPDATE agents 
            SET is_active = 1 
            WHERE is_active IS NULL
        ''')
        
        # Commit changes
        conn.commit()
        
        print("✓ Successfully added is_active column")
        print("✓ Set all existing agents as active")
        
        # Verify the migration
        cursor.execute("SELECT COUNT(*) FROM agents WHERE is_active = 1")
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} agents are active")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def main():
    """Main migration function"""
    print("Agent Active Field Migration Script")
    print("=" * 50)
    
    success = migrate_agent_active()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("\nAgents can now be soft deleted using the is_active field.")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()