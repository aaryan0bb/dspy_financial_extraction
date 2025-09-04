#!/usr/bin/env python3
"""
DSPy-Based Triplet Extraction Optimization for Financial Knowledge Graphs
========================================================================

This module demonstrates how to optimize financial document triplet extraction 
using DSPy's advanced prompt optimization techniques. It transforms raw financial 
text into structured knowledge graph entities and relationships.

Key Features:
- Multi-stage extraction pipeline
- Automatic prompt optimization using Teleprompter
- Quality assessment and validation
- Iterative refinement based on examples
"""

import json
import dspy
from dspy import InputField, OutputField, Signature
from dspy.teleprompt import BootstrapFewShot, MIPRO
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# 1. CORE SIGNATURES FOR TRIPLET EXTRACTION
# ============================================================================

class EntityExtractionSignature(Signature):
    """Extract financial entities from text with proper classification."""
    text_chunk: str = InputField(desc="Financial document text chunk to analyze")
    domain_context: str = InputField(desc="Domain context (equities, derivatives, macro, etc.)")
    entities_json: str = OutputField(desc="JSON array of entities following the specified format")


class RelationshipExtractionSignature(Signature):
    """Extract relationships between identified entities."""
    text_chunk: str = InputField(desc="Financial document text chunk")
    entities_list: str = InputField(desc="JSON list of previously extracted entities")
    relationships_json: str = OutputField(desc="JSON array of relationships following the specified format")


class ScenarioExtractionSignature(Signature):
    """Extract multi-hop causal scenarios from financial text."""
    text_chunk: str = InputField(desc="Financial document text chunk")
    entities_list: str = InputField(desc="Available entities")
    relationships_list: str = InputField(desc="Available relationships")
    scenarios_json: str = OutputField(desc="JSON array of causal scenarios if present")


class QualityAssessmentSignature(Signature):
    """Assess the quality and completeness of extracted triplets."""
    original_text: str = InputField(desc="Original source text")
    extracted_json: str = InputField(desc="Extracted entities and relationships JSON")
    quality_score: float = OutputField(desc="Quality score from 0.0 to 1.0")
    missing_elements: str = OutputField(desc="Description of missing or incorrect elements")


# ============================================================================
# 2. ENHANCED EXTRACTION MODULES
# ============================================================================

class FinancialEntityExtractor(dspy.Module):
    """Enhanced entity extraction with domain-specific knowledge."""
    
    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(EntityExtractionSignature)
        
    def forward(self, text_chunk: str, domain_context: str = "financial") -> Dict[str, Any]:
        # Enhanced prompt with domain context
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


# ============================================================================
# 3. QUALITY ASSESSMENT AND VALIDATION
# ============================================================================

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
# 4. COMPLETE EXTRACTION PIPELINE
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
# 5. PROMPT OPTIMIZATION WITH TELEPROMPTER
# ============================================================================

class TripletExtractionOptimizer:
    """Optimize triplet extraction using DSPy's Teleprompter."""
    
    def __init__(self, model_name: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        # Configure DSPy
        self.lm = dspy.LM(model_name)
        dspy.configure(lm=self.lm)
        
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
    
    def optimize_with_mipro(self, training_examples: List[dspy.Example],
                           validation_examples: List[dspy.Example]) -> 'OptimizedTripletExtractor':
        """Optimize using MIPRO (Multi-Prompt Instruction Optimization)."""
        
        def advanced_metric(example, pred, trace=None):
            """Advanced metric considering multiple aspects."""
            if not isinstance(pred, dict):
                return 0.0
            
            scores = []
            
            # Entity quality
            entities = pred.get("entities", [])
            entity_score = sum(1 for e in entities if e.get("properties", {}) and 
                             e.get("type") in ["Company", "Event", "Factor", "Instrument", "Metric"]) / max(len(entities), 1)
            scores.append(entity_score)
            
            # Relationship quality  
            relationships = pred.get("relationships", [])
            rel_score = sum(1 for r in relationships if r.get("properties", {}).get("confidence", 0) >= 60) / max(len(relationships), 1)
            scores.append(rel_score)
            
            # Quantitative capture
            text = example.text.lower()
            has_numbers = bool(re.search(r'\d+\.?\d*\s*(%|bp|usd|\$|yen|¥)', text))
            captured_numbers = any(
                r.get("properties", {}).get("impact_value") is not None or
                r.get("properties", {}).get("probability") is not None
                for r in relationships
            )
            quant_score = 1.0 if (not has_numbers or captured_numbers) else 0.0
            scores.append(quant_score)
            
            return sum(scores) / len(scores)
        
        # MIPRO optimization
        teleprompter = MIPRO(metric=advanced_metric, num_candidates=10, init_temperature=1.0)
        self.optimized_extractor = teleprompter.compile(self.base_extractor, 
                                                       trainset=training_examples,
                                                       valset=validation_examples)
        
        return self.optimized_extractor
    
    def extract_with_optimization(self, text: str) -> Dict[str, Any]:
        """Extract using the optimized model."""
        if self.optimized_extractor is None:
            raise ValueError("No optimized extractor available. Run optimization first.")
        
        return self.optimized_extractor(text)


# ============================================================================
# 6. DEMONSTRATION FUNCTIONS
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
                    "super_class": "Instrument",
                    "brief": "US stock market index",
                    "description": "Major US equity index reaching record highs",
                    "properties": {"ticker": "SPX"}
                },
                {
                    "name": "Macro Funds",
                    "type": "Actor", 
                    "super_class": "Actor",
                    "brief": "Hedge funds with macro strategies",
                    "description": "Funds adjusting regional equity exposure",
                    "properties": {}
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
        },
        # Additional examples...
    ]
    
    return texts, gold_extractions


def demonstrate_optimization():
    """Demonstrate the complete optimization process."""
    
    print("🚀 DSPy Triplet Extraction Optimization Demo")
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


def process_nomura_document(file_path: str, optimizer: TripletExtractionOptimizer) -> Dict[str, Any]:
    """Process the Nomura document using optimized extraction."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into chunks for processing
    chunks = [content[i:i+2000] for i in range(0, len(content), 1500)]  # Overlapping chunks
    
    all_entities = []
    all_relationships = []
    all_scenarios = []
    
    print(f"📄 Processing {len(chunks)} chunks from Nomura document...")
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 100:  # Skip very short chunks
            continue
            
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        try:
            result = optimizer.extract_with_optimization(chunk)
            
            if "error" not in result:
                all_entities.extend(result.get("entities", []))
                all_relationships.extend(result.get("relationships", []))
                all_scenarios.extend(result.get("scenarios", []))
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    # Deduplicate entities by name
    unique_entities = {}
    for entity in all_entities:
        name = entity.get("name", "")
        if name and name not in unique_entities:
            unique_entities[name] = entity
    
    final_result = {
        "entities": list(unique_entities.values()),
        "relationships": all_relationships,
        "scenarios": all_scenarios,
        "metadata": {
            "total_chunks_processed": len(chunks),
            "extraction_timestamp": datetime.now().isoformat(),
            "unique_entities": len(unique_entities),
            "total_relationships": len(all_relationships),
            "total_scenarios": len(all_scenarios)
        }
    }
    
    return final_result


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Demonstrate optimization
    optimizer = demonstrate_optimization()
    
    # Process Nomura document
    nomura_file = "graph_structures/chunking_strategies/inputs/Nomura_Nomura Quant Insights_20250707_enriched.txt"
    
    try:
        result = process_nomura_document(nomura_file, optimizer)
        
        # Save results
        output_file = "nomura_optimized_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Extraction complete! Results saved to {output_file}")
        print(f"📈 Extracted: {len(result['entities'])} entities, {len(result['relationships'])} relationships")
        
    except FileNotFoundError:
        print(f"❌ File not found: {nomura_file}")
        print("Please ensure the file path is correct.") 