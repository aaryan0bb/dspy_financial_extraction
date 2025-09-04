#!/usr/bin/env python3
"""
MIPRO Semantic Pipeline (Pydantic) – entities, relationships, scenarios
======================================================================
Uses DSPy with a Pydantic output schema so the extractor returns a
`KnowledgeGraph` object instead of raw JSON.
"""

import json, os, logging
from pathlib import Path
from typing import List, Dict, Any, ClassVar, Optional
from datetime import datetime

import dspy
from dspy.teleprompt import MIPROv2
from pydantic import ValidationError
import litellm

from src.config import configure_llm
from src.signatures import TripletExtractionSignature
from src.extractors import KnowledgeGraph, PydanticTripletExtractor
from src.metrics import semantic_similarity_metric

litellm.drop_params = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PydanticMIPRO:
    def __init__(self, few_shots_dir: str, out_dir: str):
        self.dir = Path(few_shots_dir)
        self.out = Path(out_dir)
        self.out.mkdir(exist_ok=True, parents=True)
        self.extractor = PydanticTripletExtractor()
        self.optimized = None

    def _load_examples(self):
        examples = []
        count = 0
        for p in sorted(self.dir.glob("few_shot_*/triplets.json")):
            if count >= 30:  # Load only the first 30 examples
                break
            try:
                chunk = (p.parent / "chunk.txt").read_text()
                expected_raw = p.read_text()

                # Validate gold JSON against Pydantic schema
                try:
                    _ = KnowledgeGraph.model_validate_json(expected_raw)
                except ValidationError as ve:
                    logger.warning(f"Skipping {p.parent.name}: gold JSON fails schema → {ve.errors()[0]['msg']}")
                    continue

                ex = dspy.Example(text_chunk=chunk, expected_triplets=expected_raw).with_inputs("text_chunk")
                examples.append(ex)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping {p.parent.name} due to error: {e}")
        return examples

    def run(self):
        configure_llm()
        exs = self._load_examples()
        val = exs[-5:]
        train = exs
        
        # Optimize
        mipro = MIPROv2(
            metric=semantic_similarity_metric, 
            auto="heavy", 
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            teacher_settings={"instructions": [TripletExtractionSignature.instructions]}
        )
        self.optimized = mipro.compile(self.extractor, trainset=train, valset=val)

        # Evaluate
        score = dspy.Evaluate(devset=val, metric=semantic_similarity_metric)(self.optimized)
        logger.info("Final score %.3f", score)

        # Save program and extras
        self._save_program()
        self._save_metadata(score, len(train), len(val))
        self._save_demos()
        self._save_validation_outputs(val)

    def _save_program(self):
        prog_dir = self.out / "optimized_program"
        prog_dir.mkdir(exist_ok=True)
        self.optimized.save(prog_dir, save_program=True)
        logger.info("📦 Program saved → %s", prog_dir)

    def _save_metadata(self, score: float, n_train: int, n_val: int):
        meta = {
            "timestamp": datetime.now().isoformat(),
            "score": float(score),
            "train_examples": n_train,
            "val_examples": n_val,
            "model": "gpt-4o-mini",
            "optimization": "MIPRO (pydantic heavy)"
        }
        (self.out / "results.json").write_text(json.dumps(meta, indent=2))

    def _save_demos(self):
        demos_to_save = []
        if hasattr(self.optimized, "demos") and self.optimized.demos:
            for d in self.optimized.demos:
                demo_dict = {
                    "text_chunk": d.text_chunk,
                    "expected_triplets": json.loads(d.expected_triplets)
                }
                demos_to_save.append(demo_dict)

        (self.out / "best_demos.json").write_text(json.dumps(demos_to_save, indent=2))
        logger.info("📄 Saved %d demos", len(demos_to_save))

    def _save_validation_outputs(self, val_examples):
        val_dir = self.out / "validation_runs"
        val_dir.mkdir(exist_ok=True)
        best_score, best_path = -1.0, None

        for idx, ex in enumerate(val_examples, 1):
            pred = self.optimized(text_chunk=ex.text_chunk)
            try:
                pred_json = pred.model_dump() if isinstance(pred, KnowledgeGraph) else pred
            except Exception:
                pred_json = {}

            s = semantic_similarity_metric(ex, pred_json)
            ex_dir = val_dir / f"example_{idx}"
            ex_dir.mkdir(exist_ok=True)
            (ex_dir / "chunk.txt").write_text(ex.text_chunk)
            (ex_dir / "gold.json").write_text(ex.expected_triplets)
            (ex_dir / "pred.json").write_text(json.dumps(pred_json, indent=2))
            (ex_dir / "score.txt").write_text(str(s))

            if s > best_score:
                best_score = s
                best_path = ex_dir

        if best_path:
            logger.info(f"🏅 Best validation score {best_score:.3f} at {best_path}")


def extract_knowledge_graph(text_chunk: str) -> KnowledgeGraph:
    """Helper function for direct extraction."""
    extractor = PydanticTripletExtractor()
    return extractor.forward(text_chunk)


if __name__ == "__main__":
    SHOTS_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/new_few_shots"
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/pydantic_results_simpler"
    PydanticMIPRO(SHOTS_DIR, OUTPUT_DIR).run()