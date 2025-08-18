#!/usr/bin/env python3
"""
Database migration script for separating Judge configurations from LLM configurations.
This script:
1. Creates the new judge_configurations table
2. Migrates existing LLM configs that are marked as judges to the new table
3. Updates existing agent_versions to reference judge configs instead of LLM configs for judges
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine, Base
from app.models import LLMConfiguration, JudgeConfiguration, AgentVersion
from sqlalchemy.orm import Session

def run_migration():
    """Run the judge configuration migration"""
    print("Starting judge configuration migration...")
    
    # Create all tables (this will create the new judge_configurations table)
    Base.metadata.create_all(bind=engine)
    print("✓ Created judge_configurations table")
    
    db = SessionLocal()
    try:
        # Step 1: Find all LLM configurations that have judge settings
        llm_configs_with_judge_settings = db.query(LLMConfiguration).filter(
            (LLMConfiguration.is_default_judge == True) | 
            (LLMConfiguration.default_judge_prompt.isnot(None))
        ).all()
        
        print(f"Found {len(llm_configs_with_judge_settings)} LLM configs with judge settings")
        
        # Step 2: Create corresponding judge configurations
        judge_config_mapping = {}  # old_llm_config_id -> new_judge_config_id
        
        for llm_config in llm_configs_with_judge_settings:
            # Create judge configuration
            judge_config = JudgeConfiguration(
                name=f"{llm_config.name} Judge",
                description=f"Judge configuration based on {llm_config.name}",
                base_llm_config_id=llm_config.id,
                judge_prompt=llm_config.default_judge_prompt or """You are an expert evaluator assessing the quality and correctness of AI responses.

Please evaluate the given response to the question on the following criteria:
1. Accuracy: Is the response factually correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-written and understandable?
4. Relevance: Does it directly address what was asked?

Provide a score from 0 to 1 (where 1 is excellent and 0 is poor) and explain your reasoning.""",
                llm_as_judge_threshold=llm_config.llm_as_judge_threshold,
                faithfulness_threshold=llm_config.faithfulness_threshold,
                is_default_judge=llm_config.is_default_judge
            )
            
            db.add(judge_config)
            db.flush()  # Get the ID
            
            judge_config_mapping[llm_config.id] = judge_config.id
            print(f"✓ Created judge config '{judge_config.name}' (ID: {judge_config.id})")
        
        # Step 3: Update agent versions that use these LLM configs as judges
        agent_versions = db.query(AgentVersion).filter(
            AgentVersion.judge_config_id.in_(list(judge_config_mapping.keys()))
        ).all()
        
        print(f"Found {len(agent_versions)} agent versions to update")
        
        for agent_version in agent_versions:
            old_judge_llm_id = agent_version.judge_config_id
            new_judge_config_id = judge_config_mapping.get(old_judge_llm_id)
            
            if new_judge_config_id:
                agent_version.judge_config_id = new_judge_config_id
                print(f"✓ Updated agent version {agent_version.id} judge config: {old_judge_llm_id} -> {new_judge_config_id}")
        
        # Step 4: Remove judge-specific fields from LLM configurations
        # Note: We keep the fields for now but set them to default values
        # In a production scenario, you might want to drop these columns entirely
        
        db.query(LLMConfiguration).update({
            'is_default_judge': False,
            'default_judge_prompt': None
        })
        
        print("✓ Cleaned up judge settings from LLM configurations")
        
        # Commit all changes
        db.commit()
        print("✓ Migration completed successfully!")
        
        # Summary
        print(f"\nMigration Summary:")
        print(f"- Created {len(judge_config_mapping)} judge configurations")
        print(f"- Updated {len(agent_versions)} agent versions")
        print(f"- Cleaned up LLM configuration judge settings")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    run_migration()