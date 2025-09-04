"""
DSPy Financial Extraction - Core Modules
========================================

This package contains the core modules for financial triplet extraction:
- Configuration and LLM setup
- DSPy signatures for extraction tasks
- Extractor classes with different capabilities
- Evaluation metrics for quality assessment
"""

from .config import configure_llm, configure_openai_direct, get_openai_key
from .signatures import (
    TripletExtractionSignature,
    EntityExtractionSignature, 
    RelationshipExtractionSignature,
    ScenarioExtractionSignature,
    QualityAssessmentSignature
)
from .extractors import (
    TripletExtractor,
    PydanticTripletExtractor,
    FinancialEntityExtractor,
    FinancialRelationshipExtractor,
    Entity,
    Relationship,
    Scenario,
    KnowledgeGraph
)
from .metrics import (
    triplet_metric,
    semantic_similarity_metric,
    llm_judge_metric
)

__version__ = "1.0.0"
__author__ = "Financial AI Research Team"

__all__ = [
    # Configuration
    "configure_llm",
    "configure_openai_direct", 
    "get_openai_key",
    
    # Signatures
    "TripletExtractionSignature",
    "EntityExtractionSignature",
    "RelationshipExtractionSignature", 
    "ScenarioExtractionSignature",
    "QualityAssessmentSignature",
    
    # Extractors
    "TripletExtractor",
    "PydanticTripletExtractor",
    "FinancialEntityExtractor",
    "FinancialRelationshipExtractor",
    
    # Pydantic Models
    "Entity",
    "Relationship", 
    "Scenario",
    "KnowledgeGraph",
    
    # Metrics
    "triplet_metric",
    "semantic_similarity_metric",
    "llm_judge_metric",
]