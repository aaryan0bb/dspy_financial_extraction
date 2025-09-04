#!/usr/bin/env python3
"""
Basic MIPRO Optimization Pipeline
=================================

Simple MIPRO optimization for financial triplet extraction using 
basic evaluation metrics and straightforward configuration.
"""

import json
import os
import dspy
from dspy.teleprompt import MIPROv2
from typing import List, Dict, Any
from datetime import datetime
import logging

from src.config import configure_llm
from src.signatures import TripletExtractionSignature
from src.extractors import TripletExtractor
from src.metrics import triplet_metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FewShotDataLoader:
    """Load few-shot examples for training."""
    
    def __init__(self, few_shots_dir: str):
        self.few_shots_dir = few_shots_dir
        
    def load_examples(self) -> List[dspy.Example]:
        """Load all few-shot examples."""
        examples = []
        
        for i in range(1, 21):  # Load up to 20 examples
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


class BasicMIPROPipeline:
    """Basic MIPRO optimization pipeline."""
    
    def __init__(self, few_shots_dir: str = "few_shots", output_dir: str = "basic_mipro_results"):
        self.few_shots_dir = few_shots_dir
        self.output_dir = output_dir
        self.data_loader = FewShotDataLoader(few_shots_dir)
        self.extractor = TripletExtractor()
        self.optimized_extractor = None
        
        print(f"📂 INPUT: {os.path.abspath(self.few_shots_dir)}")
        print(f"📁 OUTPUT: {os.path.abspath(self.output_dir)}")
        
    def run(self):
        """Run the complete basic MIPRO pipeline."""
        logger.info("🚀 Starting Basic MIPRO Pipeline")
        
        # 1. Configure LLM
        configure_llm()
        
        # 2. Load data
        examples = self.data_loader.load_examples()
        if not examples:
            logger.error("❌ No examples loaded!")
            return None
        
        # 3. Split data
        train_size = max(1, int(0.8 * len(examples)))
        train_examples = examples[:train_size]
        val_examples = examples[train_size:] if len(examples) > train_size else examples[:1]
        
        logger.info(f"📊 Train: {len(train_examples)}, Val: {len(val_examples)}")
        
        # 4. Optimize with MIPRO (basic settings)
        logger.info("🎯 Starting basic MIPRO optimization...")
        
        mipro = MIPROv2(
            metric=triplet_metric,
            auto="light",  # Use light auto mode for basic optimization
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        
        self.optimized_extractor = mipro.compile(
            self.extractor,
            trainset=train_examples,
            valset=val_examples,
            requires_permission_to_run=False
        )
        
        logger.info("✅ Basic MIPRO optimization completed!")
        
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
        self._save_results(score, train_examples, val_examples)
        
        return {
            "optimized_extractor": self.optimized_extractor,
            "score": score,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples)
        }
    
    def _save_results(self, score: float, train_examples: List[dspy.Example], val_examples: List[dspy.Example]):
        """Save optimization results."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "optimization_method": "MIPRO Basic",
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "score": score,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "auto_mode": "light"
        }
        
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📝 Results saved to {self.output_dir}/")


if __name__ == "__main__":
    INPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/few_shots"
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/basic_mipro_results"
    
    pipeline = BasicMIPROPipeline(INPUT_DIR, OUTPUT_DIR)
    results = pipeline.run()
    
    if results:
        print(f"\n✅ Basic MIPRO Pipeline completed!")
        print(f"📈 Final Score: {results['score']:.3f}")
        print(f"📊 Examples: {results['train_examples']} train, {results['val_examples']} val")
        print(f"💾 Results: {OUTPUT_DIR}/")
    else:
        print("❌ Pipeline failed!")