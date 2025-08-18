#!/usr/bin/env python3
"""
Simple database migration script for judge configurations.
This script just creates the table structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine, Base
from app.models import LLMConfiguration, JudgeConfiguration

def run_migration():
    """Run the judge configuration migration"""
    print("Starting simple judge configuration migration...")
    
    # Create all tables (this will create the new judge_configurations table)
    Base.metadata.create_all(bind=engine)
    print("✓ Created judge_configurations table")
    
    db = SessionLocal()
    try:
        # Create a default judge configuration if we have LLM configs
        llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
        
        if llm_configs:
            # Use the first LLM config as the base for a default judge
            base_llm = llm_configs[0]
            
            judge_config = JudgeConfiguration(
                name="Default Judge",
                description="Default judge configuration",
                base_llm_config_id=base_llm.id,
                judge_prompt="""You are an expert evaluator assessing the quality and correctness of AI responses.

Please evaluate the given response to the question on the following criteria:
1. Accuracy: Is the response factually correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-written and understandable?
4. Relevance: Does it directly address what was asked?

Provide a score from 0 to 1 (where 1 is excellent and 0 is poor) and explain your reasoning.""",
                llm_as_judge_threshold=0.8,
                faithfulness_threshold=0.8,
                is_default_judge=True
            )
            
            db.add(judge_config)
            db.commit()
            print(f"✓ Created default judge configuration based on '{base_llm.name}'")
        else:
            print("! No LLM configurations found, no default judge created")
        
        print("✓ Migration completed successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    run_migration()