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

class Run(Base):
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    agent_version_id = Column(Integer, ForeignKey("agent_versions.id"), nullable=False)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    
    # Run status and metadata
    status = Column(String, default="pending")  # pending, running, completed, failed
    total_test_cases = Column(Integer, default=0)
    completed_test_cases = Column(Integer, default=0)
    
    # Aggregate results
    overall_score = Column(Float)
    pass_rate = Column(Float)
    aggregate_results = Column(JSON)  # {metric_name: {score: float, passed: bool}}
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Export artifacts
    csv_export_path = Column(String)  # Path to CSV export
    json_export_path = Column(String)  # Path to JSON export
    html_report_path = Column(String)  # Path to HTML report
    
    # Relationships
    agent_version = relationship("AgentVersion")
    dataset_version = relationship("DatasetVersion")
    test_cases = relationship("TestCase", back_populates="run", cascade="all, delete-orphan")

class TestCase(Base):
    __tablename__ = "test_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    
    # Test case data
    case_id = Column(String, nullable=False)  # From dataset
    question = Column(Text, nullable=False)
    context = Column(Text)  # Optional, from dataset
    expected_answer = Column(Text)  # Optional, from dataset
    actual_answer = Column(Text, nullable=False)  # From uploaded answers
    
    # Overall score for this test case
    overall_score = Column(Float)
    passed = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="test_cases")
    metric_results = relationship("MetricResult", back_populates="test_case", cascade="all, delete-orphan")

class MetricResult(Base):
    __tablename__ = "metric_results"
    
    id = Column(Integer, primary_key=True, index=True)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False)
    
    # Metric details
    metric_name = Column(String, nullable=False)  # llm_as_judge, faithfulness
    score = Column(Float, nullable=False)  # 0.0 to 1.0
    passed = Column(Boolean, nullable=False)  # Based on threshold
    threshold = Column(Float, nullable=False)  # Threshold used
    
    # Additional metadata
    reasoning = Column(Text)  # Judge reasoning if available
    execution_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test_case = relationship("TestCase", back_populates="metric_results")