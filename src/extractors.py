#!/usr/bin/env python3
"""
Financial Triplet Extractors
============================

This module contains the core extractor classes used across different
training and inference pipelines.
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

import dspy
from pydantic import BaseModel, Field

from .signatures import TripletExtractionSignature, EntityExtractionSignature, RelationshipExtractionSignature

logger = logging.getLogger(__name__)

# Pydantic Models for structured extraction
class Entity(BaseModel):
    name: str = Field(...)
    type: str = Field(...)
    brief: str = Field(...)
    description: str = Field(...)
    properties: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class Scenario(BaseModel):
    scenario_id: str
    root_trigger: str
    steps: List[Dict[str, Any]]

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    scenarios: List[Scenario] = []


class TripletExtractor(dspy.Module):
    """Basic triplet extractor using DSPy ChainOfThought."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(TripletExtractionSignature)
        
    def forward(self, text_chunk: str) -> Dict[str, Any]:
        """Extract triplets from text chunk."""
        try:
            result = self.extract(text_chunk=text_chunk)
            
            # Parse JSON output
            triplets_data = json.loads(result.triplets_json)
            triplets_data["success"] = True
            
            return triplets_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {
                "success": False,
                "error": f"JSON decode error: {e}",
                "entities": [],
                "relationships": [],
                "scenarios": []
            }
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "relationships": [],
                "scenarios": []
            }


class PydanticTripletExtractor(dspy.Module):
    """Pydantic-validated triplet extractor."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(TripletExtractionSignature)
        
    def forward(self, text_chunk: str) -> KnowledgeGraph:
        """Extract and validate triplets using Pydantic."""
        result = self.extract(text_chunk=text_chunk)
        
        try:
            # Parse JSON and validate with Pydantic
            raw_data = json.loads(result.triplets_json)
            return KnowledgeGraph(**raw_data)
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            return KnowledgeGraph(entities=[], relationships=[], scenarios=[])


class FinancialEntityExtractor(dspy.Module):
    """Enhanced entity extraction with domain-specific knowledge."""
    
    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(EntityExtractionSignature)
        
    def forward(self, text_chunk: str, domain_context: str = "financial") -> Dict[str, Any]:
        enhanced_prompt = self._build_entity_prompt()
        
        result = self.extract_entities(
            text_chunk=f"{enhanced_prompt}\n\nText to analyze:\n{text_chunk}",
            domain_context=domain_context
        )
        
        try:
            entities = json.loads(result.entities_json)
            return {"entities": entities, "success": True}
        except json.JSONDecodeError as e:
            return {"entities": [], "success": False, "error": str(e)}
    
    def _build_entity_prompt(self) -> str:
        return """
You are a senior financial risk analyst extracting entities for a knowledge graph.

ENTITY EXTRACTION RULES:
1. Focus on MATERIAL entities with quantitative or qualitative importance
2. Canonicalize names (e.g., "GOOG" vs "Alphabet Inc." → same Company)
3. Include confidence ≥ 60 only
4. Prioritize entities with numeric values, probabilities, or risk implications

ENTITY TYPES:
- Company: Include "ticker" when available
- Event: Include "date" and "event_type"  
- Metric: Include "unit" and "as_of"
- Factor: Market forces or conditions
- Instrument: Financial products or securities

OUTPUT FORMAT: Valid JSON array only.
"""


class FinancialRelationshipExtractor(dspy.Module):
    """Enhanced relationship extraction with financial semantics."""
    
    def __init__(self):
        super().__init__()
        self.extract_relationships = dspy.ChainOfThought(RelationshipExtractionSignature)
        
    def forward(self, text_chunk: str, entities: List[Dict]) -> Dict[str, Any]:
        entities_str = json.dumps(entities, indent=2)
        enhanced_prompt = self._build_relationship_prompt()
        
        result = self.extract_relationships(
            text_chunk=f"{enhanced_prompt}\n\nText:\n{text_chunk}",
            entities_list=entities_str
        )
        
        try:
            relationships = json.loads(result.relationships_json)
            return {"relationships": relationships, "success": True}
        except json.JSONDecodeError as e:
            return {"relationships": [], "success": False, "error": str(e)}
    
    def _build_relationship_prompt(self) -> str:
        return """
Extract financial relationships with QUANTITATIVE FOCUS.

RELATIONSHIP TYPES:
- EXPOSED_TO: Risk exposure (with probability/impact_value)
- CAUSES: Direct causation (with lag_days)
- TRIGGERS: Event triggering (with probability)
- IMPACTS: Financial impact (with impact_value/unit)
- OWNS/HOLDS: Ownership/position (with quantities)

REQUIRED PROPERTIES:
- probability: 0-1 or null
- lag_days: integer or null  
- impact_value: number or null
- unit: "USD", "%", "bp", etc.
- confidence: 0-100

Only include relationships with confidence ≥ 60.
Capture ALL numeric magnitudes from text.
"""