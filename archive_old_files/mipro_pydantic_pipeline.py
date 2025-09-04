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

import dspy
from pydantic import BaseModel, Field, ValidationError
from dspy.teleprompt import MIPROv2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import litellm
litellm.drop_params = True
# ─────────────────── Logging ───────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────── LLM CONFIG ───────────────────
OPENAI_API_KEY = ""  # <-- put your key here


def configure_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY env var")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini",temperature=1.0,max_tokens=20000))
    litellm.drop_params = True # Ensure this is set when LLM is configured
    logger.info("✅ LLM ready (gpt-4o-mini)")

# ─────────────────── Pydantic schema ───────────────────
class Entity(BaseModel):
    name: str = Field(...)
    type: str = Field(...)
    brief: str = Field(...) # ≤15 words essence
    description: str = Field(...)
    properties: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    description: str # ≤20 words summary of the relationship        
    properties: Dict[str, Any] = Field(default_factory=dict)

class Scenario(BaseModel):
    scenario_id: str
    root_trigger: str
    steps: List[Dict[str, Any]]

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    scenarios: List[Scenario] = []

class ExtractGraphTriplet(dspy.Signature):
    text_chunk: str = dspy.InputField(desc="Paragraph to extract KG from")
    knowledge: KnowledgeGraph = dspy.OutputField(desc="Structured KG")
    
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

Return JSON only.
"""

# ─────────────────── Triplet Extractor ───────────────────
class TripletExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractGraphTriplet)

    def forward(self, text_chunk: str) -> KnowledgeGraph:
        result = self.extract(text_chunk=text_chunk)
        return result.knowledge

# ─────────────────── Semantic metrics ───────────────────
MODEL_EMB = SentenceTransformer("all-MiniLM-L6-v2")


def _sent_sim(a: List[str], b: List[str], th: float = 0.55) -> float:
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


def metric(example, prediction, trace=None):
    # Convert to dicts (using model_dump for Pydantic V2 compatibility)
    pred = prediction.model_dump() if isinstance(prediction, KnowledgeGraph) else prediction
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

# ---------- helpers ----------

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

# ─────────────────── Pipeline ───────────────────
class PydanticMIPRO:
    def __init__(self, few_shots_dir: str, out_dir: str):
        self.dir = Path(few_shots_dir)
        self.out = Path(out_dir)
        self.out.mkdir(exist_ok=True, parents=True)
        self.extractor = TripletExtractor()
        self.optimized = None

    def _load_examples(self):
        examples = []
        count = 0
        for p in sorted(self.dir.glob("few_shot_*/triplets.json")):
            if count >= 30: # Load only the first 10 examples
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
        # Optimise
        mipro = MIPROv2(metric=metric, auto="heavy", max_bootstrapped_demos=8,max_labeled_demos=8,teacher_settings={"instructions": [ExtractGraphTriplet.instructions]})
        self.optimized = mipro.compile(self.extractor, trainset=train, valset=val)

        # Evaluate
        score = dspy.Evaluate(devset=val, metric=metric)(self.optimized)
        logger.info("Final score %.3f", score)

        # Save program and extras
        self._save_program()
        self._save_metadata(score, len(train), len(val))
        self._save_demos()
        self._save_validation_outputs(val)

    # ───────────── saving helpers ─────────────
    def _save_program(self):
        prog_dir = self.out / "optimized_program"
        prog_dir.mkdir(exist_ok=True)
        self.optimized.save(prog_dir, save_program=True)
        logger.info("📦 Program saved → %s", prog_dir)

    def _save_metadata(self, score: float, n_train: int, n_val: int):
        meta = {
            "timestamp": datetime.now().isoformat(),
            "score": float(score), # Ensure score is a standard Python float
            "train_examples": n_train,
            "val_examples": n_val,
            "model": "o4-mini",
            "optimization": "MIPRO (pydantic heavy)"
        }
        (self.out / "results.json").write_text(json.dumps(meta, indent=2))

    def _save_demos(self):
        demos_to_save = []
        # Collect demos from the top-level optimized program
        if hasattr(self.optimized, "demos") and self.optimized.demos:
            for d in self.optimized.demos:
                # Assuming d is a dspy.Example object with text_chunk and expected_triplets
                # expected_triplets is already a JSON string, so no need to dump Pydantic model
                demo_dict = {
                    "text_chunk": d.text_chunk,
                    "expected_triplets": json.loads(d.expected_triplets) # Parse the string to a dict for consistency
                }
                demos_to_save.append(demo_dict)

        # Collect demos from predictors within the optimized program (if any and if structured differently)
        # This part might need further refinement based on dspy's internal demo structure for predictors
        # For now, if the problem is just with the main demos, the above should suffice.

        (self.out / "best_demos.json").write_text(json.dumps(demos_to_save, indent=2))
        logger.info("📄 Saved %d demos", len(demos_to_save))

    # ---------------- validation outputs ----------------
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

            s = metric(ex, pred_json)
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

# ─────────────────── CLI entry ───────────────────
if __name__ == "__main__":
    SHOTS_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/new_few_shots"               # hard-coded input directory
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/pydantic_results_simpler"        # hard-coded output directory
    PydanticMIPRO(SHOTS_DIR, OUTPUT_DIR).run()