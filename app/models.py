from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
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