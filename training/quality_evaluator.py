#!/usr/bin/env python3
"""
LLM Judge Quality Evaluator
===========================

This pipeline uses an LLM (GPT-4) as a judge to evaluate the quality
of triplet extractions, providing more nuanced assessment than
simple similarity metrics.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

from openai import OpenAI
import dspy
from dspy.teleprompt import MIPROv2

from src.config import configure_llm, get_openai_key
from src.signatures import TripletExtractionSignature
from src.extractors import TripletExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Judge configuration
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_SYSTEM_PROMPT = """
You are a world-class financial knowledge-graph evaluator. You will receive:

• GOLD JSON: with "entities", "relationships", and optional "scenarios".  
• PREDICTED JSON: same schema.

Evaluate the prediction on:

**Accuracy & Completeness**: which entities/relations are correct, missing, or spurious?  
**Semantic Faithfulness**: does the prediction capture the *meaning* and nuance of the gold?  
**Content Richness**: are scenarios or contextual details present when expected?  
**Clarity & Structure**: is the JSON well-organized and unambiguous?  
**No Hallucinations**: does it avoid introducing unsupported facts?

Return ONLY a single numeric score from 0.0 to 1.0 (where 1.0 = perfect match).
"""

_JUDGE_CACHE: Dict[Tuple[str,str], float] = {}


def call_judge_llm(gold_json: str, pred_json: str) -> float:
    """Query GPT-4 judge and parse score."""
    key = (gold_json, pred_json)
    if key in _JUDGE_CACHE:
        return _JUDGE_CACHE[key]

    try:
        client = OpenAI(api_key=get_openai_key())
        
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "GOLD JSON:\n" + gold_json.strip() + "\n\n" +
                    "PREDICTED JSON:\n" + pred_json.strip() + "\n\n" +
                    "Score:"
                ),
            },
        ]
        
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=4,
        )
        text = resp.choices[0].message.content.strip()
        
        # Extract score
        import re
        match = re.search(r"[01](?:\.\d+)?", text)
        score = float(match.group(0)) if match else 0.0
        
        logger.info(f"✅ Judge score: {score:.3f}")
        
    except Exception as e:
        logger.error(f"Judge LLM error: {e}")
        score = 0.0

    _JUDGE_CACHE[key] = score
    return score


def llm_judge_metric(example, prediction, trace=None) -> float:
    """Metric wrapper for DSPy using LLM judge."""
    try:
        if isinstance(prediction, str):
            pred_json = prediction.strip()
        elif isinstance(prediction, dict):
            pred_json = json.dumps(prediction, separators=(",", ":"))
        else:
            pred_json = str(prediction)

        if not pred_json:
            return 0.0

        gold_json = getattr(example, "expected_triplets", "").strip()
        if not gold_json:
            return 0.0

        # Get score from LLM judge
        score = call_judge_llm(gold_json, pred_json)

        if trace and hasattr(trace, "__dict__"):
            trace.score = score
        return score
        
    except Exception as e:
        logger.error(f"Metric wrapper error: {e}")
        return 0.0


class LLMJudgeEvaluator:
    """Pipeline using LLM judge for quality evaluation."""
    
    def __init__(self, few_shots_dir: str = "few_shots", output_dir: str = "llm_judge_results"):
        self.few_shots_dir = few_shots_dir
        self.output_dir = output_dir
        self.extractor = TripletExtractor()
        self.optimized_extractor = None
        
        print(f"📂 INPUT: {os.path.abspath(self.few_shots_dir)}")
        print(f"📁 OUTPUT: {os.path.abspath(self.output_dir)}")

    def load_examples(self) -> List[dspy.Example]:
        """Load few-shot examples."""
        examples: List[dspy.Example] = []
        i = 1
        while True:
            example_dir = os.path.join(self.few_shots_dir, f"few_shot_{i}")
            if not os.path.exists(example_dir):
                break
            try:
                with open(os.path.join(example_dir, "chunk.txt"), "r", encoding="utf-8") as f:
                    text_chunk = f.read().strip()
                with open(os.path.join(example_dir, "triplets.json"), "r", encoding="utf-8") as f:
                    expected_triplets = json.dumps(json.load(f))
                
                ex = dspy.Example(
                    text_chunk=text_chunk, 
                    expected_triplets=expected_triplets
                ).with_inputs("text_chunk")
                examples.append(ex)
                logger.info(f"Loaded {example_dir}")
            except Exception as e:
                logger.error(f"Failed to load {example_dir}: {e}")
            i += 1
            if i > 50:  # Safety limit
                break
        
        logger.info(f"Total examples: {len(examples)}")
        return examples

    def run(self):
        """Run the LLM judge evaluation pipeline."""
        logger.info("🚀 LLM Judge Evaluation Pipeline starting...")
        
        # Configure LLM
        configure_llm()
        
        # Load examples
        examples = self.load_examples()
        if len(examples) < 2:
            logger.error("Need at least 2 examples")
            return

        # Split data
        num_val = min(5, len(examples)//2)
        train_examples = examples
        val_examples = examples[-num_val:]
        logger.info(f"Train {len(train_examples)} | Val {len(val_examples)}")

        # Run MIPRO with LLM judge
        mipro = MIPROv2(
            metric=llm_judge_metric,
            auto="medium",
            max_bootstrapped_demos=4,
            teacher_settings={"instructions": [TripletExtractionSignature.instructions]},
        )

        self.optimized_extractor = mipro.compile(
            self.extractor,
            trainset=train_examples,
            valset=val_examples,
            requires_permission_to_run=False,
        )
        logger.info("✅ Optimization done")

        # Evaluate
        evaluate = dspy.Evaluate(devset=val_examples, metric=llm_judge_metric, display_progress=True)
        score = evaluate(self.optimized_extractor)
        logger.info(f"Final LLM judge score: {score:.3f}")

        # Save results
        self._save_results(score, train_examples, val_examples)

    def _save_results(self, score: float, train_examples: List[dspy.Example], val_examples: List[dspy.Example]):
        """Save evaluation results."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        metadata = {
            "method": "MIPRO-LLM-Judge",
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "judge_model": JUDGE_MODEL,
            "score": float(score),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
        }
        
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved → {self.output_dir}")


if __name__ == "__main__":
    INPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/few_shots"
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/llm_judge_results"
    
    evaluator = LLMJudgeEvaluator(INPUT_DIR, OUTPUT_DIR)
    evaluator.run()