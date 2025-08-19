from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import hashlib
import json

from app.database import Base

class LLMConfiguration(Base):
    __tablename__ = "llm_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    
    # LLM Provider Settings
    connector_type = Column(String, nullable=False)  # litellm_openai, litellm_anthropic, etc.
    endpoint_url = Column(String, nullable=False)
    api_key = Column(String)  # Encrypted in production
    model_name = Column(String, nullable=False)
    
    # Model Parameters
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)
    timeout_seconds = Column(Integer, default=30)
    rate_limit_per_minute = Column(Integer, default=60)
    
    # Additional provider-specific parameters
    additional_params = Column(JSON)  # For provider-specific configurations
    
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def display_provider(self):
        """Get user-friendly provider name"""
        provider_map = {
            "litellm_openai": "OpenAI",
            "litellm_anthropic": "Anthropic", 
            "litellm_google": "Google",
            "litellm_aws": "AWS Bedrock",
            "litellm_azure": "Azure OpenAI",
            "litellm_ollama": "Ollama",
            "litellm_huggingface": "HuggingFace",
            "openai": "OpenAI (Direct)",
            "http": "Custom HTTP"
        }
        return provider_map.get(self.connector_type, self.connector_type)
    
    def to_connector_config(self):
        """Convert to ConnectorConfig object"""
        from app.connectors import ConnectorConfig
        return ConnectorConfig(
            connector_type=self.connector_type,
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            rate_limit_per_minute=self.rate_limit_per_minute,
            additional_params=self.additional_params
        )

class JudgeConfiguration(Base):
    __tablename__ = "judge_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    
    # Base LLM Configuration reference
    base_llm_config_id = Column(Integer, ForeignKey("llm_configurations.id"), nullable=False)
    
    # Judge profile: simple, detailed, custom
    judge_profile = Column(String, nullable=False, default="simple")
    
    # Judge-specific settings
    judge_prompt = Column(Text, nullable=False)
    llm_as_judge_threshold = Column(Float, default=0.8)
    # faithfulness metrics removed for simplicity
    
    # Judge model parameters (overrides base config if provided)
    judge_temperature = Column(Float)  # Override temperature for judge
    judge_max_tokens = Column(Integer)  # Override max_tokens for judge
    
    # Status and metadata
    is_default_judge = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    base_llm_config = relationship("LLMConfiguration", foreign_keys=[base_llm_config_id])
    
    def to_connector_config(self):
        """Convert to ConnectorConfig object using base LLM config with judge overrides"""
        from app.connectors import ConnectorConfig
        return ConnectorConfig(
            connector_type=self.base_llm_config.connector_type,
            endpoint_url=self.base_llm_config.endpoint_url,
            api_key=self.base_llm_config.api_key,
            model_name=self.base_llm_config.model_name,
            temperature=self.judge_temperature if self.judge_temperature is not None else self.base_llm_config.temperature,
            max_tokens=self.judge_max_tokens if self.judge_max_tokens is not None else self.base_llm_config.max_tokens,
            timeout_seconds=self.base_llm_config.timeout_seconds,
            rate_limit_per_minute=self.base_llm_config.rate_limit_per_minute,
            additional_params=self.base_llm_config.additional_params
        )

# Reference datasets are now linked to specific agent versions
class ReferenceDataset(Base):
    __tablename__ = "reference_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_version_id = Column(Integer, ForeignKey("agent_versions.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    row_count = Column(Integer, nullable=False)
    content_hash = Column(String, nullable=False)  # SHA256 of normalized content
    file_path = Column(String, nullable=False)  # Path to stored JSONL file
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to agent version
    agent_version = relationship("AgentVersion", back_populates="reference_datasets")
    
    @staticmethod
    def generate_content_hash(rows):
        """Generate SHA256 hash of normalized row data"""
        content = json.dumps(rows, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode()).hexdigest()

# Legacy models kept for backward compatibility during migration
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
    is_active = Column(Boolean, default=True)
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
    
    # LLM Configuration references
    llm_config_id = Column(Integer, ForeignKey("llm_configurations.id"), nullable=True)  # For chatbot inference
    judge_config_id = Column(Integer, ForeignKey("judge_configurations.id"), nullable=True)  # For evaluation judge
    
    # Legacy model configuration (kept for backward compatibility)
    model_config = Column(JSON, nullable=True)  # {provider, model, temperature, max_tokens, etc}
    
    # LLM-as-judge is the only evaluation metric now
    # faithfulness and other metrics removed for simplicity
    
    # Default thresholds
    default_thresholds = Column(JSON)  # {metric_name: threshold_value}
    
    # Legacy judge model configuration (kept for backward compatibility)
    judge_model_config = Column(JSON)  # {provider, model, temperature, etc}
    
    # Editable judge prompt (overrides LLM config default if provided)
    judge_prompt = Column(Text)
    
    # Connector functionality removed - CSV upload only
    
    # Evaluation configuration
    store_verbose_artifacts = Column(Boolean, default=False)  # Store judge prompts/responses
    
    # Relationships
    agent = relationship("Agent", back_populates="versions")
    llm_config = relationship("LLMConfiguration", foreign_keys=[llm_config_id])
    judge_config = relationship("JudgeConfiguration", foreign_keys=[judge_config_id])
    reference_datasets = relationship("ReferenceDataset", back_populates="agent_version", cascade="all, delete-orphan")
    runs = relationship("Run", foreign_keys="Run.agent_version_id", back_populates="agent_version")
    
    @property
    def display_name(self):
        return f"v{self.version_number}"

class Run(Base):
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    agent_version_id = Column(Integer, ForeignKey("agent_versions.id"), nullable=False)
    reference_dataset_id = Column(Integer, ForeignKey("reference_datasets.id"), nullable=True)  # Nullable for backward compatibility
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)  # Legacy field
    
    # Run status and metadata
    status = Column(String, default="pending")  # pending, running, completed, failed
    evaluation_source = Column(String, default="upload")  # upload, connector
    total_test_cases = Column(Integer, default=0)
    completed_test_cases = Column(Integer, default=0)
    
    # Aggregate results
    overall_score = Column(Float)
    pass_rate = Column(Float)
    aggregate_results = Column(JSON)  # {metric_name: {score: float, passed: bool}}
    
    # Threshold overrides for re-scoring (Phase 8)
    threshold_overrides = Column(JSON)  # {metric_name: threshold_value} - overrides agent defaults
    
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
    reference_dataset = relationship("ReferenceDataset")
    dataset_version = relationship("DatasetVersion")  # Legacy field
    test_cases = relationship("TestCase", back_populates="run", cascade="all, delete-orphan")
    
    @property
    def dataset(self):
        """Get the dataset for this run (new or legacy)"""
        return self.reference_dataset or self.dataset_version

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
    
    # Verbose artifacts (optional, for debugging/transparency)
    raw_judge_response = Column(Text)  # Full raw response from judge
    judge_prompt_used = Column(Text)  # Actual prompt sent to judge
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test_case = relationship("TestCase", back_populates="metric_results")