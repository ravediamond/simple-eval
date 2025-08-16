from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import hashlib
import json

from app.database import Base

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to versions
    versions = relationship("DatasetVersion", back_populates="dataset", order_by="desc(DatasetVersion.version_number)")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    notes = Column(Text)
    row_count = Column(Integer, nullable=False)
    content_hash = Column(String, nullable=False)  # SHA256 of normalized content
    file_path = Column(String, nullable=False)  # Path to stored JSONL file
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to dataset
    dataset = relationship("Dataset", back_populates="versions")
    
    @staticmethod
    def generate_content_hash(rows):
        """Generate SHA256 hash of normalized row data"""
        content = json.dumps(rows, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    def display_name(self):
        return f"v{self.version_number}"

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    tags = Column(JSON)  # List of strings
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to versions
    versions = relationship("AgentVersion", back_populates="agent", order_by="desc(AgentVersion.version_number)")

class AgentVersion(Base):
    __tablename__ = "agent_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Model configuration for the tested model
    model_config = Column(JSON, nullable=False)  # {provider, model, temperature, max_tokens, etc}
    
    # Metric toggles
    llm_as_judge_enabled = Column(Boolean, default=True)
    faithfulness_enabled = Column(Boolean, default=True)
    
    # Default thresholds
    default_thresholds = Column(JSON)  # {metric_name: threshold_value}
    
    # Judge model configuration
    judge_model_config = Column(JSON)  # {provider, model, temperature, etc}
    
    # Editable judge prompt
    judge_prompt = Column(Text)
    
    # Relationship to agent
    agent = relationship("Agent", back_populates="versions")
    
    @property
    def display_name(self):
        return f"v{self.version_number}"