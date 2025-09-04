#!/usr/bin/env python3
"""
Bootstrap Few-Shot Trainer for Financial Triplet Extraction
===========================================================

This module demonstrates how to optimize financial document triplet extraction 
using DSPy's Bootstrap Few-Shot optimization. It transforms raw financial 
text into structured knowledge graph entities and relationships.

Key Features:
- Multi-stage extraction pipeline
- Bootstrap Few-Shot optimization
- Quality assessment and validation
- Iterative refinement based on examples
"""

import json
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from datetime import datetime
import logging

from src.config import configure_llm
from src.signatures import (
    EntityExtractionSignature,
    RelationshipExtractionSignature, 
    ScenarioExtractionSignature,
    QualityAssessmentSignature
)
from src.extractors import FinancialEntityExtractor, FinancialRelationshipExtractor

logger = logging.getLogger(__name__)


# ============================================================================
# SCENARIO AND QUALITY EXTRACTION MODULES  
# ============================================================================

class ScenarioExtractor(dspy.Module):
    """Extract multi-hop causal scenarios for complex risk chains."""
    
    def __init__(self):
        super().__init__()
        self.extract_scenarios = dspy.ChainOfThought(ScenarioExtractionSignature)
        
    def forward(self, text_chunk: str, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        entities_str = json.dumps([e.get("name", "") for e in entities])
        rels_str = json.dumps([f"{r.get('source', '')} -[{r.get('rel_type', '')}]-> {r.get('target', '')}" 
                              for r in relationships])
        
        result = self.extract_scenarios(
            text_chunk=text_chunk,
            entities_list=entities_str,
            relationships_list=rels_str
        )
        
        try:
            scenarios = json.loads(result.scenarios_json)
            return {"scenarios": scenarios, "success": True}
        except json.JSONDecodeError as e:
            return {"scenarios": [], "success": False, "error": str(e)}


class QualityAssessor(dspy.Module):
    """Assess extraction quality and suggest improvements."""
    
    def __init__(self):
        super().__init__()
        self.assess_quality = dspy.ChainOfThought(QualityAssessmentSignature)
        
    def forward(self, original_text: str, extracted_data: Dict) -> Dict[str, Any]:
        extracted_json = json.dumps(extracted_data, indent=2)
        
        result = self.assess_quality(
            original_text=original_text,
            extracted_json=extracted_json
        )
        
        return {
            "quality_score": result.quality_score,
            "missing_elements": result.missing_elements,
            "needs_refinement": result.quality_score < 0.8
        }


# ============================================================================
# COMPLETE EXTRACTION PIPELINE
# ============================================================================

class OptimizedTripletExtractor(dspy.Module):
    """Complete optimized pipeline for financial triplet extraction."""
    
    def __init__(self):
        super().__init__()
        self.entity_extractor = FinancialEntityExtractor()
        self.relationship_extractor = FinancialRelationshipExtractor()
        self.scenario_extractor = ScenarioExtractor()
        self.quality_assessor = QualityAssessor()
        
    def forward(self, text: str, domain_context: str = "financial") -> Dict[str, Any]:
        """Main extraction pipeline with quality control."""
        
        # Step 1: Extract entities
        entity_result = self.entity_extractor(text, domain_context)
        if not entity_result["success"]:
            return {"error": f"Entity extraction failed: {entity_result.get('error', '')}"}
        
        entities = entity_result["entities"]
        
        # Step 2: Extract relationships
        rel_result = self.relationship_extractor(text, entities)
        if not rel_result["success"]:
            return {"error": f"Relationship extraction failed: {rel_result.get('error', '')}"}
        
        relationships = rel_result["relationships"]
        
        # Step 3: Extract scenarios (if applicable)
        scenario_result = self.scenario_extractor(text, entities, relationships)
        scenarios = scenario_result.get("scenarios", []) if scenario_result["success"] else []
        
        # Step 4: Quality assessment
        extracted_data = {
            "entities": entities,
            "relationships": relationships,
            "scenarios": scenarios
        }
        
        quality_result = self.quality_assessor(text, extracted_data)
        
        return {
            **extracted_data,
            "quality_assessment": quality_result,
            "metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "scenario_count": len(scenarios)
            }
        }


# ============================================================================
# PROMPT OPTIMIZATION WITH BOOTSTRAP FEW-SHOT
# ============================================================================

class TripletExtractionOptimizer:
    """Optimize triplet extraction using DSPy's Bootstrap Few-Shot."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Configure DSPy
        configure_llm(model=model_name)
        
        # Initialize modules
        self.base_extractor = OptimizedTripletExtractor()
        self.optimized_extractor = None
        
    def create_training_examples(self, text_samples: List[str], 
                               gold_extractions: List[Dict]) -> List[dspy.Example]:
        """Create training examples for optimization."""
        examples = []
        
        for text, gold in zip(text_samples, gold_extractions):
            example = dspy.Example(
                text=text,
                domain_context="financial",
                expected_entities=gold.get("entities", []),
                expected_relationships=gold.get("relationships", []),
                expected_scenarios=gold.get("scenarios", [])
            ).with_inputs("text", "domain_context")
            
            examples.append(example)
            
        return examples
    
    def optimize_with_bootstrap(self, training_examples: List[dspy.Example], 
                               max_bootstrapped_demos: int = 8) -> 'OptimizedTripletExtractor':
        """Optimize using Bootstrap Few-Shot."""
        
        def triplet_extraction_metric(example, pred, trace=None):
            """Custom metric for triplet extraction quality."""
            if not isinstance(pred, dict) or "entities" not in pred:
                return 0.0
            
            # Entity coverage score
            expected_entities = set(e.get("name", "") for e in example.expected_entities)
            extracted_entities = set(e.get("name", "") for e in pred.get("entities", []))
            
            entity_recall = len(expected_entities & extracted_entities) / max(len(expected_entities), 1)
            entity_precision = len(expected_entities & extracted_entities) / max(len(extracted_entities), 1)
            
            # Relationship coverage score  
            expected_rels = set(f"{r.get('source', '')}-{r.get('rel_type', '')}-{r.get('target', '')}" 
                              for r in example.expected_relationships)
            extracted_rels = set(f"{r.get('source', '')}-{r.get('rel_type', '')}-{r.get('target', '')}" 
                               for r in pred.get("relationships", []))
            
            rel_recall = len(expected_rels & extracted_rels) / max(len(expected_rels), 1)
            
            # Quality assessment score
            quality_score = pred.get("quality_assessment", {}).get("quality_score", 0.0)
            
            # Combined score
            return 0.4 * entity_recall + 0.2 * entity_precision + 0.3 * rel_recall + 0.1 * quality_score
        
        # Bootstrap optimization
        teleprompter = BootstrapFewShot(metric=triplet_extraction_metric, 
                                       max_bootstrapped_demos=max_bootstrapped_demos)
        
        self.optimized_extractor = teleprompter.compile(self.base_extractor, trainset=training_examples)
        
        return self.optimized_extractor
    
    def extract_with_optimization(self, text: str) -> Dict[str, Any]:
        """Extract using the optimized model."""
        if self.optimized_extractor is None:
            raise ValueError("No optimized extractor available. Run optimization first.")
        
        return self.optimized_extractor(text)


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def create_sample_training_data() -> tuple:
    """Create sample training data for demonstration."""
    
    # Sample text chunks
    texts = [
        """The S&P 500 rose to a fresh record high last week, lifted in part by speculators 
        covering short positions. Macro funds' long exposure to European equities and short 
        exposure to US equities have both been walked back almost all of the way to the neutral line.""",
        
        """CTAs added to their long exposure, albeit only slightly. We expect CTAs to continue 
        buying the dip for as long as the index stays above 39,000. Their short exposure is now 
        only about 30% of what it was at the recent peak.""",
        
        """The VIX call skew is at the 94th percentile, indicating elevated hedging demand. 
        Leveraged VIX ETN AUM has reached multi-year highs, suggesting potential for a volatility squeeze."""
    ]
    
    # Corresponding gold standard extractions
    gold_extractions = [
        {
            "entities": [
                {
                    "name": "S&P 500",
                    "type": "Instrument",
                    "brief": "US stock market index",
                    "description": "Major US equity index reaching record highs",
                    "properties": {"ticker": "SPX"}
                }
            ],
            "relationships": [
                {
                    "source": "Speculators",
                    "target": "S&P 500", 
                    "rel_type": "IMPACTS",
                    "properties": {
                        "confidence": 75,
                        "impact_value": 1.0,
                        "unit": "direction"
                    },
                    "description": "Short covering lifted index higher"
                }
            ]
        }
    ]
    
    return texts, gold_extractions


def demonstrate_optimization():
    """Demonstrate the complete optimization process."""
    
    print("🚀 Bootstrap Few-Shot Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = TripletExtractionOptimizer()
    
    # Create training data
    texts, gold_extractions = create_sample_training_data()
    training_examples = optimizer.create_training_examples(texts, gold_extractions)
    
    print(f"📚 Created {len(training_examples)} training examples")
    
    # Test base extraction
    print("\n🔍 Testing base extraction:")
    base_result = optimizer.base_extractor(texts[0])
    print(f"Base extraction found {len(base_result.get('entities', []))} entities")
    
    # Optimize with Bootstrap
    print("\n🎯 Optimizing with Bootstrap Few-Shot...")
    optimized_extractor = optimizer.optimize_with_bootstrap(training_examples)
    
    # Test optimized extraction
    print("\n✨ Testing optimized extraction:")
    optimized_result = optimizer.extract_with_optimization(texts[0])
    print(f"Optimized extraction found {len(optimized_result.get('entities', []))} entities")
    
    # Compare results
    print("\n📊 Comparison:")
    print(f"Base Quality Score: {base_result.get('quality_assessment', {}).get('quality_score', 'N/A')}")
    print(f"Optimized Quality Score: {optimized_result.get('quality_assessment', {}).get('quality_score', 'N/A')}")
    
    return optimizer


if __name__ == "__main__":
    # Demonstrate optimization
    optimizer = demonstrate_optimization()
    print("\n🎉 Bootstrap Few-Shot optimization completed!")