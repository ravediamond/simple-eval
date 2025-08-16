#!/usr/bin/env python3
"""
Database migration script to add missing columns.
Run this when the database schema is outdated.
"""

import sqlite3
import os
from pathlib import Path

def get_db_path():
    """Get the database path from environment or use default"""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/simple_eval.db")
    if db_url.startswith("sqlite:///"):
        return db_url[10:]  # Remove "sqlite:///" prefix
    return "./data/simple_eval.db"

def migrate_database():
    """Apply database migrations"""
    db_path = get_db_path()
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("🔄 Checking database schema...")
        
        # Check if evaluation_source column exists in runs table
        cursor.execute("PRAGMA table_info(runs)")
        columns = [col[1] for col in cursor.fetchall()]
        
        migrations_applied = []
        
        # Migration 1: Add evaluation_source column
        if 'evaluation_source' not in columns:
            print("Adding evaluation_source column to runs table...")
            cursor.execute('ALTER TABLE runs ADD COLUMN evaluation_source VARCHAR DEFAULT "upload"')
            migrations_applied.append("evaluation_source column")
        
        # Migration 1b: Add export path columns to runs table
        if 'csv_export_path' not in columns:
            print("Adding csv_export_path column to runs table...")
            cursor.execute('ALTER TABLE runs ADD COLUMN csv_export_path VARCHAR')
            migrations_applied.append("csv_export_path column")
        
        if 'json_export_path' not in columns:
            print("Adding json_export_path column to runs table...")
            cursor.execute('ALTER TABLE runs ADD COLUMN json_export_path VARCHAR')
            migrations_applied.append("json_export_path column")
        
        if 'html_report_path' not in columns:
            print("Adding html_report_path column to runs table...")
            cursor.execute('ALTER TABLE runs ADD COLUMN html_report_path VARCHAR')
            migrations_applied.append("html_report_path column")
        
        # Migration 2: Add store_verbose_artifacts column to agent_versions
        cursor.execute("PRAGMA table_info(agent_versions)")
        agent_columns = [col[1] for col in cursor.fetchall()]
        
        if 'store_verbose_artifacts' not in agent_columns:
            print("Adding store_verbose_artifacts column to agent_versions table...")
            cursor.execute('ALTER TABLE agent_versions ADD COLUMN store_verbose_artifacts BOOLEAN DEFAULT 0')
            migrations_applied.append("store_verbose_artifacts column")
        
        # Migration 3: Add verbose artifact columns to metric_results
        cursor.execute("PRAGMA table_info(metric_results)")
        metric_columns = [col[1] for col in cursor.fetchall()]
        
        if 'raw_judge_response' not in metric_columns:
            print("Adding raw_judge_response column to metric_results table...")
            cursor.execute('ALTER TABLE metric_results ADD COLUMN raw_judge_response TEXT')
            migrations_applied.append("raw_judge_response column")
        
        if 'judge_prompt_used' not in metric_columns:
            print("Adding judge_prompt_used column to metric_results table...")
            cursor.execute('ALTER TABLE metric_results ADD COLUMN judge_prompt_used TEXT')
            migrations_applied.append("judge_prompt_used column")
        
        # Migration 4: Add threshold_overrides column to runs table (Phase 8)
        cursor.execute("PRAGMA table_info(runs)")
        runs_columns = [col[1] for col in cursor.fetchall()]
        
        if 'threshold_overrides' not in runs_columns:
            print("Adding threshold_overrides column to runs table...")
            cursor.execute('ALTER TABLE runs ADD COLUMN threshold_overrides JSON')
            migrations_applied.append("threshold_overrides column")
        
        # Commit all changes
        conn.commit()
        
        if migrations_applied:
            print(f"✅ Successfully applied {len(migrations_applied)} migrations:")
            for migration in migrations_applied:
                print(f"   - {migration}")
        else:
            print("✅ Database schema is up to date!")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def main():
    """Main migration function"""
    print("🗄️  Simple Eval Database Migration")
    print("=" * 40)
    
    try:
        migrate_database()
        print("\n🎉 Migration completed successfully!")
        print("You can now start the application with:")
        print("python -m uvicorn app.main:app --reload --port 8000")
        
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        print("Please check the error and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())