#!/usr/bin/env python3
"""
Evaluation Metrics for Financial Triplet Extraction
===================================================

This module contains various evaluation metrics used to assess
the quality of triplet extraction results.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Global sentence transformer model for semantic metrics
MODEL_EMB = SentenceTransformer("all-MiniLM-L6-v2")


def triplet_metric(example, prediction, trace=None) -> float:
    """Simple metric to evaluate triplet extraction quality."""
    
    # Check if prediction succeeded
    if isinstance(prediction, dict) and not prediction.get("success", True):
        return 0.0
    
    try:
        # Parse expected and predicted data
        if hasattr(example, 'expected_triplets'):
            expected_data = json.loads(example.expected_triplets)
        else:
            return 0.0
        
        if isinstance(prediction, str):
            pred_data = json.loads(prediction)
        elif hasattr(prediction, 'model_dump'):
            pred_data = prediction.model_dump()
        else:
            pred_data = prediction
        
        expected_entities = expected_data.get("entities", [])
        expected_relationships = expected_data.get("relationships", [])
        
        pred_entities = pred_data.get("entities", [])
        pred_relationships = pred_data.get("relationships", [])
        
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
        logger.error(f"Metric calculation error: {e}")
        return 0.0


def semantic_similarity_metric(example, prediction, trace=None) -> float:
    """Semantic similarity metric using sentence transformers."""
    
    try:
        # Convert to dicts
        pred = prediction.model_dump() if hasattr(prediction, 'model_dump') else prediction
        gold = json.loads(example.expected_triplets)

        ent_score = _sent_sim(
            [e["name"] for e in pred.get("entities", [])],
            [e["name"] for e in gold.get("entities", [])],
        )
        rel_score = _sent_sim(
            [enhanced_to_sentence(r) for r in pred.get("relationships", [])],
            [enhanced_to_sentence(r) for r in gold.get("relationships", [])],
        )
        
        def _sc_to_para(sc: Dict) -> str:
            root = sc.get("root_trigger", "")
            parts: List[str] = []
            for st in sc.get("steps", []):
                src = st.get("source", "")
                tgt = st.get("target", "")
                rel = st.get("edge_type", "").lower()
                lag = st.get("lag_days")
                sent = f"{src} {rel} {tgt}"
                if lag not in (None, "", 0):
                    sent += f" after {lag} days"
                parts.append(sent)
            if root:
                parts.insert(0, f"Scenario starts with {root}")
            return " ; ".join(parts)

        scen_score = _sent_sim(
            [_sc_to_para(s) for s in pred.get("scenarios", [])],
            [_sc_to_para(s) for s in gold.get("scenarios", [])],
            th=0.6,
        )
        return 0.6 * rel_score + 0.3 * ent_score + 0.1 * scen_score
        
    except Exception as e:
        logger.error(f"Semantic metric error: {e}")
        return 0.0


def _sent_sim(a: List[str], b: List[str], th: float = 0.55) -> float:
    """Calculate semantic similarity between two lists of strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    E = MODEL_EMB.encode(a + b, normalize_embeddings=True)
    sim = cosine_similarity(E[: len(a)], E[len(a) :])
    row, col = linear_sum_assignment(-sim)
    tp, tot = 0, 0.0
    for i, j in zip(row, col):
        if sim[i, j] >= th:
            tp += 1
            tot += sim[i, j]
    prec = tp / len(a)
    rec = tp / len(b)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    avg = tot / tp if tp else 0
    return 0.7 * f1 + 0.3 * avg


def enhanced_to_sentence(rel: Dict) -> str:
    """Convert relationship dict to a natural sentence for embedding."""
    src = rel.get("source", "")
    tgt = rel.get("target", "")
    rel_type = rel.get("type", "").upper()
    desc = rel.get("description", "")

    tpl = {
        "CAUSES": f"{src} causes {tgt}",
        "IMPACTS": f"{src} impacts {tgt}",
        "OWNS": f"{src} owns {tgt}",
        "HOLDS": f"{src} holds {tgt}",
        "EXPOSED_TO": f"{src} is exposed to {tgt}",
        "TRIGGERS": f"{src} triggers {tgt}",
    }
    base = tpl.get(rel_type, f"{src} {rel_type.lower()} {tgt}")
    return f"{base}. {desc}" if desc else base


def llm_judge_metric(example, prediction, trace=None) -> float:
    """LLM-based evaluation metric (placeholder for actual implementation)."""
    # This would be implemented with actual LLM judge calls
    # For now, fall back to simple metric
    return triplet_metric(example, prediction, trace)