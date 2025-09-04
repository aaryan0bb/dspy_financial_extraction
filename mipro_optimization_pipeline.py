#!/usr/bin/env python3
"""
MIPRO Optimization Pipeline for Financial Triplet Extraction
==========================================================

This pipeline uses DSPy's MIPRO (Multi-Prompt Instruction Optimization) 
with your golden few-shot examples to optimize triplet extraction using OpenAI GPT-4o-mini.

Following the simple DSPy tutorial pattern.
"""

import json
import os
import dspy
from dspy import InputField, OutputField, Signature
from dspy.teleprompt import MIPROv2
from typing import List, Dict, Any, Tuple, ClassVar
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. SIMPLE SIGNATURE FOR TRIPLET EXTRACTION
# ============================================================================

class TripletExtractionSignature(dspy.Signature):
    """Extract financial entities and relationships from text as JSON."""
    
    text_chunk: str = InputField(desc="Financial document text chunk to analyze")
    triplets_json: str = OutputField(
        desc="JSON with entities, relationships, and scenarios",
        prefix="JSON:\n"
    )
    
    instructions: ClassVar[str] = """
You are a senior financial risk analyst turning research‑document paragraphs
and tables into a KNOWLEDGE GRAPH that supports multi‑hop, quantitative
risk reasoning while remaining compact and low‑noise.

   • Output **valid JSON** only with three top‑level keys:
     1. "entities":   list of nodes (see ENTITY FORMAT)
     2. "relationships": list of edges (see REL FORMAT)
     3. "scenarios":  OPTIONAL list of ordered causal chains when text
                      describes knock‑on effects (see SCENARIO FORMAT)

══════════════════════════════════════════════════════════════
ENTITY FORMAT  (strict keys, no extras)
{
  "name":            "<canonical name>",
  "type":            "<Company | Event | Factor | Instrument | Metric | Other>",
  "super_class":     "<Actor | Event | Factor | Instrument | Metric>",
  "custom_type":     "<only if type=='Other', else null>",
  "type_explanation":"<if custom_type, ≤15 words>",
  "brief":           "<≤15 words essence>",
  "description":     "<≤40 words, why material>",
  "properties":      { key:value ...   // numbers stay numbers }
}
Important conventions
•  Company → properties MUST include "ticker" when available.
•  Event  → include "date" and "event_type".
•  Metric → include "unit" ("USD", "%", "bp", etc.) and "as_of".

══════════════════════════════════════════════════════════════
REL FORMAT
{
  "source":          "<entity name>",
  "target":          "<entity name>",
  "rel_type":        "<EXPOSED_TO | CAUSES | TRIGGERS | IMPACTS | OWNS | HOLDS | Other>",
  "custom_rel_type": "<only if rel_type=='Other', else null>",
  "rel_explanation": "<if custom_rel_type, ≤15 words>",
  "properties": {
        "probability" : <0‑1|null>,
        "lag_days"    : <int|null>,
        "impact_value": <number|null>,
        "unit"        : "<unit|null>",
        "non_linear"  : <true|false|null>,
        "confidence"  : <0‑100>
  },
  "description":     "<≤20 words>"
}

Extraction rules
•  **Must capture numeric magnitude** whenever the sentence/table gives it
   (%, bp, $mm, σ).  Keep raw numbers.
•  **Materiality gate**: ignore relationships with no quantitative or
   qualitative importance (e.g., routine disclosures, boilerplate).
•  **Canonicalise** duplicates: “GOOG” vs “Alphabet Inc.” → same Company node.
•  Only include relationships whose confidence ≥ 60.

══════════════════════════════════════════════════════════════
SCENARIO FORMAT  (use when document narrates a chain)
{
  "scenario_id":    "<slug>",
  "root_trigger":   "<root Event or Factor name>",
  "probability":    <0‑1 or null>,
  "steps": [
      { "hop":1, "edge_type":"CAUSES", "source":"<name>", "target":"<name>", 
        "lag_days":0,   "prob":1.0 },
      { "hop":2, "edge_type":"IMPACTS", "source":"<name>", "target":"<name>",
        "lag_days":14,  "prob":0.85, "impact_value":-22, "unit":"%" },
      ...
  ]
}

══════════  MATERIALITY HOOKS  ══════════
KEEP a candidate entity/relationship **only if** it satisfies **one or more**:
  • quantitative figure present (%, USD, bp, σ, etc.)
  • explicit probability / confidence / likelihood language
  • legal / regulatory / catastrophic risk mention
  • appears at least twice within the chunk
Otherwise discard.

══════════ EXTRACTION WORKFLOW (internal) ══════════
1. Draft unlimited candidates.
2. Apply MATERIALITY HOOKS.
3. Rank remaining by importance to investors.
4. Output top entities (≤15) and relationships (≤25).

Return JSON only."""


# ============================================================================
# 2. SIMPLE LLM CONFIGURATION
# ============================================================================

OPENAI_API_KEY = "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"  # Replace with your actual API key

def configure_llm():
    """Configure DSPy to use OpenAI."""
    import os
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    
    logger.info("✅ OpenAI configured with gpt-4o-mini")
    return lm


# ============================================================================
# 3. SIMPLE DATA LOADER
# ============================================================================

class FewShotDataLoader:
    """Load few-shot examples for training."""
    
    def __init__(self, few_shots_dir: str):
        self.few_shots_dir = few_shots_dir
        
    def load_examples(self) -> List[dspy.Example]:
        """Load all few-shot examples."""
        examples = []
        
        for i in range(1, 6):  # few_shot_1 to few_shot_5
            example_dir = os.path.join(self.few_shots_dir, f"few_shot_{i}")
            
            if not os.path.exists(example_dir):
                continue
                
            try:
                # Load text chunk
                chunk_file = os.path.join(example_dir, "chunk.txt")
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    text_chunk = f.read().strip()
                
                # Load expected triplets
                triplets_file = os.path.join(example_dir, "triplets.json")
                with open(triplets_file, 'r', encoding='utf-8') as f:
                    expected_triplets = json.dumps(json.load(f))
                
                # Create DSPy example
                example = dspy.Example(
                    text_chunk=text_chunk,
                    expected_triplets=expected_triplets
                ).with_inputs("text_chunk")
                
                examples.append(example)
                logger.info(f"✅ Loaded few_shot_{i}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load few_shot_{i}: {e}")
                continue
        
        logger.info(f"📚 Loaded {len(examples)} examples total")
        return examples


# ============================================================================
# 4. SIMPLE TRIPLET EXTRACTOR
# ============================================================================

class TripletExtractor(dspy.Module):
    """Simple triplet extractor module."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(TripletExtractionSignature)
        
    def forward(self, text_chunk: str) -> Dict[str, Any]:
        """Extract triplets from text."""
        try:
            result = self.extract(text_chunk=text_chunk)
            
            # Parse JSON output
            triplets_data = json.loads(result.triplets_json)
            triplets_data["success"] = True
            
            return triplets_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return {
                "success": False,
                "error": f"JSON decode error: {e}",
                "entities": [],
                "relationships": [],
                "scenarios": []
            }
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "relationships": [],
                "scenarios": []
            }


# ============================================================================
# 5. SIMPLE EVALUATION METRIC
# ============================================================================

def triplet_metric(example, prediction, trace=None):
    """Simple metric to evaluate triplet extraction quality."""
    
    # Check if prediction succeeded
    if not prediction.get("success", False):
        return 0.0
    
    try:
        # Parse expected and predicted data
        expected_data = json.loads(example.expected_triplets)
        
        expected_entities = expected_data.get("entities", [])
        expected_relationships = expected_data.get("relationships", [])
        
        pred_entities = prediction.get("entities", [])
        pred_relationships = prediction.get("relationships", [])
        
        # Simple entity matching (by name)
        expected_entity_names = {e.get("name", "").lower() for e in expected_entities}
        pred_entity_names = {e.get("name", "").lower() for e in pred_entities}
        
        entity_overlap = len(expected_entity_names & pred_entity_names)
        entity_recall = entity_overlap / max(len(expected_entity_names), 1)
        entity_precision = entity_overlap / max(len(pred_entity_names), 1)
        entity_f1 = 2 * entity_recall * entity_precision / max(entity_recall + entity_precision, 0.001)
        
        # Simple relationship matching (by source-target pair)
        expected_rel_pairs = {
            f"{r.get('source', '').lower()}-{r.get('target', '').lower()}"
            for r in expected_relationships
        }
        pred_rel_pairs = {
            f"{r.get('source', '').lower()}-{r.get('target', '').lower()}"
            for r in pred_relationships
        }
        
        rel_overlap = len(expected_rel_pairs & pred_rel_pairs)
        rel_recall = rel_overlap / max(len(expected_rel_pairs), 1)
        rel_precision = rel_overlap / max(len(pred_rel_pairs), 1)
        rel_f1 = 2 * rel_recall * rel_precision / max(rel_recall + rel_precision, 0.001)
        
        # Combined score (equal weight to entities and relationships)
        final_score = 0.5 * entity_f1 + 0.5 * rel_f1
        
        return final_score
        
    except Exception as e:
        logger.error(f"❌ Metric calculation error: {e}")
        return 0.0


# ============================================================================
# 6. SIMPLE MIPRO PIPELINE
# ============================================================================

class MIPROPipeline:
    """Simple MIPRO optimization pipeline."""
    
    def __init__(self, few_shots_dir: str = "few_shots", output_dir: str = "mipro_results"):
        self.few_shots_dir = few_shots_dir
        self.output_dir = output_dir
        self.data_loader = FewShotDataLoader(few_shots_dir)
        self.extractor = TripletExtractor()
        self.optimized_extractor = None
        
        # Print paths
        print(f"📂 INPUT: {os.path.abspath(self.few_shots_dir)}")
        print(f"📁 OUTPUT: {os.path.abspath(self.output_dir)}")
        
    def run(self):
        """Run the complete MIPRO pipeline."""
        logger.info("🚀 Starting MIPRO Pipeline")
        
        # 1. Configure LLM
        configure_llm()
        
        # 2. Load data
        examples = self.data_loader.load_examples()
        if not examples:
            logger.error("❌ No examples loaded!")
            return None
        
        # 3. Split data
        train_size = max(1, int(0.7 * len(examples)))
        train_examples = examples[:train_size]
        val_examples = examples[train_size:] if len(examples) > train_size else examples[:1]
        
        logger.info(f"📊 Train: {len(train_examples)}, Val: {len(val_examples)}")
        
        # 4. Optimize with MIPRO
        logger.info("🎯 Starting MIPRO optimization...")
        
        mipro = MIPROv2(
            metric=triplet_metric,
            auto="medium"
        )
        
        self.optimized_extractor = mipro.compile(
            self.extractor,
            trainset=train_examples,
            max_bootstrapped_demos=4,
            requires_permission_to_run=False
        )
        
        logger.info("✅ MIPRO optimization completed!")
        
        # 5. Evaluate
        logger.info("📊 Evaluating optimized extractor...")
        
        evaluate = dspy.Evaluate(
            devset=val_examples,
            metric=triplet_metric,
            display_progress=True
        )
        
        score = evaluate(self.optimized_extractor)
        logger.info(f"📈 Final Score: {score:.3f}")
        
        # 6. Save results
        self._save_results(score, val_examples)
        
        return {
            "optimized_extractor": self.optimized_extractor,
            "score": score,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples)
        }
    
    def _save_results(self, score: float, val_examples: List[dspy.Example]):
        """Save optimization results."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "optimization_method": "MIPRO",
            "timestamp": datetime.now().isoformat(),
            "model": "OpenAI GPT-4o-mini",
            "score": score,
            "num_examples": len(val_examples)
        }
        
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Extract and save optimized prompt
        try:
            optimized_prompt = self._extract_optimized_prompt()
            
            with open(os.path.join(self.output_dir, "optimized_prompt.txt"), 'w') as f:
                f.write("="*60 + "\n")
                f.write("MIPRO OPTIMIZED PROMPT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Score: {score:.3f}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                f.write(optimized_prompt)
                f.write("\n" + "="*60)
            
            logger.info(f"📝 Results saved to {self.output_dir}/")
            
        except Exception as e:
            logger.warning(f"Could not extract optimized prompt: {e}")
    
    def _extract_optimized_prompt(self) -> str:
        """Extract the optimized prompt from MIPRO."""
        try:
            if self.optimized_extractor and hasattr(self.optimized_extractor, 'extract'):
                signature = self.optimized_extractor.extract.signature
                
                parts = []
                parts.append("OPTIMIZED INSTRUCTIONS:")
                parts.append(signature.instructions)
                parts.append("\nOPTIMIZED SIGNATURE:")
                parts.append(f"Input: {[f.name for f in signature.input_fields]}")
                parts.append(f"Output: {[f.name for f in signature.output_fields]}")
                
                return "\n".join(parts)
            
            return "Optimized prompt extraction not available."
            
        except Exception as e:
            return f"Prompt extraction failed: {str(e)}"


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🚀 MIPRO Optimization for Financial Triplet Extraction")
    print("=" * 60)
    
    # Configuration
    INPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/few_shots"
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/mipro_results"
    
    # Validate input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: Input directory '{INPUT_DIR}' not found!")
        return
    
    # Run pipeline
    pipeline = MIPROPipeline(INPUT_DIR, OUTPUT_DIR)
    results = pipeline.run()
    
    if results:
        print(f"\n✅ MIPRO Pipeline completed!")
        print(f"📈 Final Score: {results['score']:.3f}")
        print(f"📊 Examples: {results['train_examples']} train, {results['val_examples']} val")
        print(f"💾 Results: {OUTPUT_DIR}/")
    else:
        print("❌ Pipeline failed!")


if __name__ == "__main__":
    main() 