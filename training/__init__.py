"""
Training Pipelines for Financial Triplet Extraction
===================================================

This package contains various training and optimization pipelines:
- Bootstrap Few-Shot optimization
- MIPRO optimization with different configurations
- LLM-judge based quality evaluation
"""

from .bootstrap_trainer import TripletExtractionOptimizer, OptimizedTripletExtractor
from .mipro_trainer import PydanticMIPRO, extract_knowledge_graph
from .mipro_trainer_basic import BasicMIPROPipeline
from .quality_evaluator import LLMJudgeEvaluator, llm_judge_metric

__all__ = [
    "TripletExtractionOptimizer",
    "OptimizedTripletExtractor", 
    "PydanticMIPRO",
    "extract_knowledge_graph",
    "BasicMIPROPipeline",
    "LLMJudgeEvaluator",
    "llm_judge_metric",
]