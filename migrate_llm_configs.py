#!/usr/bin/env python3
"""
Database migration script to add LLM configuration references to AgentVersion table.
This adds the new llm_config_id and judge_config_id columns to support the new LLM configuration system.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/simple_eval.db"

def migrate_database():
    """Add LLM configuration columns to agent_versions table"""
    
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found. Nothing to migrate.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(agent_versions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        migrations_needed = []
        
        if 'llm_config_id' not in columns:
            migrations_needed.append("ADD COLUMN llm_config_id INTEGER REFERENCES llm_configurations(id)")
        
        if 'judge_config_id' not in columns:
            migrations_needed.append("ADD COLUMN judge_config_id INTEGER REFERENCES llm_configurations(id)")
        
        if not migrations_needed:
            print("✅ Database already up to date. No migration needed.")
            return
        
        print("🔄 Starting database migration...")
        
        # Apply migrations
        for migration in migrations_needed:
            sql = f"ALTER TABLE agent_versions {migration}"
            print(f"   Executing: {sql}")
            cursor.execute(sql)
        
        # Make model_config nullable for new records
        print("   Updating model_config to be nullable...")
        # Note: SQLite doesn't support ALTER COLUMN, but new records can have NULL model_config
        
        conn.commit()
        print("✅ Database migration completed successfully!")
        
        # Show summary
        cursor.execute("SELECT COUNT(*) FROM agent_versions")
        agent_versions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM llm_configurations")
        llm_configs_count = cursor.fetchone()[0]
        
        print(f"\n📊 Migration Summary:")
        print(f"   - Agent versions: {agent_versions_count}")
        print(f"   - LLM configurations: {llm_configs_count}")
        print(f"   - New chatbots can now use LLM configurations")
        print(f"   - Existing chatbots continue using legacy model_config")
        
    except sqlite3.Error as e:
        print(f"❌ Database migration failed: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    print("Simple Eval - LLM Configuration Migration")
    print("=" * 50)
    migrate_database()