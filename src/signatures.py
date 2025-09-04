#!/usr/bin/env python3
"""
DSPy Signatures for Financial Triplet Extraction
=================================================

This module contains all DSPy signature definitions used across the financial
extraction pipeline. Signatures define the input/output interface and 
instructions for language model calls.
"""

from typing import ClassVar
import dspy
from dspy import InputField, OutputField, Signature


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
•  **Canonicalise** duplicates: "GOOG" vs "Alphabet Inc." → same Company node.
•  Only include relationships whose confidence ≥ 60.

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