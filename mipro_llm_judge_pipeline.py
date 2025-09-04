#!/usr/bin/env python3
"""
MIPRO LLM-Judge Pipeline for Financial Triplet Extraction
========================================================

This pipeline mirrors `mipro_semantic_pipeline.py` but replaces the
sentence-embedding semantic metric with a Large-Language-Model (LLM)
judge.  The judge (OpenAI GPT-4o-mini) receives the expected JSON and
predicted JSON, then outputs a single score in [0,1].  That score is
used by DSPy's MIPROv2 optimizer.

⚠  Using an LLM as a metric may incur significant cost and latency
   because the judge is called many times during optimization.
   Consider reducing `max_bootstrapped_demos` or using a cheaper model
   for the judge if needed.
"""

from __future__ import annotations

import os, json, logging, time
from datetime import datetime
from typing import List, Dict, Any, Tuple, ClassVar

import openai
from openai import OpenAI # Import the new OpenAI client
import dspy
from dspy.teleprompt import MIPROv2
from dspy import InputField, OutputField

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. LLM & API CONFIGURATION
# ---------------------------------------------------------------------------

# Put your OpenAI key here or in the OPENAI_API_KEY env variable
OPENAI_API_KEY = ""  # <- replace

JUDGE_MODEL = "o3"  # Judge LLM – uses Completion endpoint
EXTRACT_MODEL_ALIAS = "openai/gpt-4o"  # Used by DSPy extractor

def configure_llm():
    """Configure DSPy extractor LLM and OpenAI creds."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-"):
        raise ValueError("Please set a valid OPENAI_API_KEY at top of script")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    lm = dspy.LM(EXTRACT_MODEL_ALIAS)
    dspy.configure(lm=lm)
    logger.info("✅ DSPy configured with %s", EXTRACT_MODEL_ALIAS)

# ---------------------------------------------------------------------------
# 2. TRIPLET EXTRACTION SIGNATURE (same as previous pipeline)
# ---------------------------------------------------------------------------

class TripletExtractionSignature(dspy.Signature):
    text_chunk: str = InputField(desc="Financial document text chunk to analyze")
    triplets_json: str = OutputField(
        desc="JSON with entities and relationships", prefix="JSON:\n"
    )
    # Reuse detailed instructions from previous file but keep minimal here
    instructions: ClassVar[str] = """Your task is to extract the node-entity triplets from text chunks <chunk> by following instructions in <triple_extraction_prompt>. To assist you with this task, I have also provided you with a sample example consisting of input chunk and resulting triplets in json format in <triplet_example> section

Please follow instructions carefully and generate triplets in json format

<triple_extraction_prompt>
You are a senior financial risk analyst turning research‑document paragraphs
and tables into a KNOWLEDGE GRAPH that supports multi‑hop, quantitative
risk reasoning.

   • Output **valid JSON** only with three top‑level keys:
     1. "entities":   list of nodes (see ENTITY FORMAT)
     2. "relationships": list of edges (see REL FORMAT)
     3. "scenarios":  OPTIONAL list of ordered causal chains when text
                      describes knock‑on effects (see SCENARIO FORMAT)

══════════════════════════════════════════════════════════════
ENTITY FORMAT  (strict keys, no extras)
{
  "name":           "<canonical name>",
  "type":           "<Company | Event | Factor | Metric | Instrument>",
  "brief":          "<≤15 words essence>",
  "description":    "<≤40 words, why material>",
  "properties":     { <key:value pairs, numbers stay numbers> }
}

Important conventions
•  Company → properties MUST include "ticker" when available.
•  Event  → include "date" (ISO‑8601) and "event_type".
•  Metric → include "unit" ("USD", "%", "bp", etc.) and "as_of".

══════════════════════════════════════════════════════════════
REL FORMAT
{
  "source":         "<entity name>",
  "target":         "<entity name>",
  "type":           "<EXPOSED_TO | CAUSES | TRIGGERS | IMPACTS | OWNS | HOLDS>",
  "properties": {
        "probability" : <0‑1 or null>,
        "lag_days"    : <integer or null>,
        "impact_value": <number or null>,
        "unit"        : "<unit or null>",
        "non_linear"  : <true|false|null>,
        "confidence"  : <0‑100>      // analyst confidence
  },
  "description":    "<≤20 words summary>"
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

══════════════════════════════════════════════════════════════
Extraction checklist (apply to *each* paragraph / table):
✓ Identify any EVENT or FACTOR with explicit or implied financial impact.  
✓ Pull adjacent quantitative metrics (magnitude, unit, date).  
✓ Detect verbs that suggest CAUSAL or IMPACT links ("drives", "leads to",
  "forces", "cuts cash‑flow", "de‑pegs").  
✓ If text narrates more than one hop, build a SCENARIO chain with ordered steps.  
✓ Discard boilerplate, accounting policy text, immaterial percentages (<1 %) 
  unless explicitly framed as risk.

Return JSON only.

</triple_extraction_prompt>


<triplet_example>
input:
| Date | Historical Weekly (lhs) $bn | Current Weekly (rhs) $bn |
    | :----- | :-------------------------- | :----------------------- |
    | 13-Jun | 1.40 | 0.60 |
    | 20-Jun | 1.35 | 0.85 |
    | 2-Jul | 1.85 | 3.90 |
    | 9-Jul | 1.10 | 2.00 |
    | 15-Jul | 1.00 | 1.50 |
    | 25-Jul | 1.25 | 1.00 |
    | 31-Jul | 0.80 | 0.80 |
    | 12-Aug | 0.55 | 0.40 |
    | 22-Aug | 0.50 | 0.30 |
    | 28-Aug | 0.75 | 0.50 |

Summary: The chart "Average Weekly Est. Retail Net Flow 2025 vs. History" displays retail equity flows in billions of dollars from mid-June to late August, comparing "Historical Weekly" (yellow line, left axis) with "Current Weekly" (blue line, right axis). A striking feature is the "Current Weekly" flow's dramatic surge to a peak of nearly $4bn around July 2nd, significantly higher than historical norms. Following this spike, current flows experience a steep decline, eventually falling below historical weekly averages by mid-July and continuing to trend lower into late August, mimicking the general seasonal slowdown. Historical flows also peak around early July (approx. $1.85bn) before gradually decreasing through August.

Explanation: This chart visually reinforces the broader market narrative regarding a significant slowdown in passive inflows. The sharp decline in "Current Weekly" retail flows after an early July surge aligns with the context that retail demand for US equities has slowed considerably, shifting from $3.5bn/day to $1.5bn/day. This trend, coupled with anticipated slowdowns in other systematic and buyback flows, indicates a potential increase in market fragility, making equities more susceptible to corrections in the coming months.

page_end

well                                                                                                                                                                                  0% 10%         20%    30%    40% 50%        60%    70%    80%     90% 100% 
could                                                                        -0.20% -0.15%     -0.10%   -0.05%   0.00%    0.05%    0.10%    0.15%     0.20%                Source: 15-Mar 7 
still           the    Content. historical            Source: 
                                                          1H Jan                                                                                                                 22-Mar - 
        highlighting,   There                             2H Jan                                                                                                                       Mar -                                      2                                                     Est. 
                                                       Morgan                                                                                                               Morgan 29 
mean    and     market is              Positioning-wise   1H Feb                                                                                                                    5-Apr- 
 that    the    is      still   medians) the             2H Feb                                                                                              Average             12-Apr                                                                                                 S&P 
                                                       Stanley 1H Mar                                                                                                       Stanley 19-Apr 
 HFs                            with    most              2H Mar                                                                                             Daily                                                                                                                       500 
                 trading,        P/L                   QDS 1H Apr                                                                                                            QDS 26-Apr 
 have            the     ample                            2H Apr                                                                                              S&P                  3-May 
  to     potentially     room                            1H May                                                                                               500                10-May                                                                                                  Market 
  take                           cushions                 2H May                                                                                                                 17-May 
                         for             supportive        1H Jun                                                                                                                24-May                                                                                                   Cap 
  nets    limited positive nets (US                        2H Jun                                                                                             Return             31-May                                                                                                   in 
                                  L/S                       1H Jul                                                                                             by                    7-Jun - 
                          to             dynamic            2H Jul                                                                                                                 14 Jun - 
  higher                  go              is               1H Aug                                                                                              Half 
          incremental             +4.9%                    2H Aug                                                                                                                  21-Jun                                                                                                 Buyback 
                  fundamental higher, YTD that             1H Sep                                                                                                                  28-Jun 
   before                                 HF               2H Sep                                                                                              Month                    5-Jul 
   the     forward trends and             nets              1H Oct                                                                                                                    12-Jul 
                                  coming                     2H Oct                                                                                            Since                 19-Jul                                                                                               Blackouts 
                   Mike                                    1H     Nov                                                                                                                26-Jul 
   market impact           arguably into   remain          2H Nov                                                                                              2005                   2-Aug - 
   can     of      Wilson          the                      1H Dec                                                                                                                    9-Aug 
    go                     nets                             2H Dec 
            tariffs and            week)   modest 
    lower. (see     team   should   per 
                                           (51%, 
            their           be      MS 
             latest have    higher PB       inline 
                    been                    with 
                            given 
             here). 
                             how 
             This

| Date | % S&P 500 Market Cap in Buyback Blackout |
    |---|---|
    | 15-Mar | 5% |
    | 29-Mar | 45% |
    | 19-Apr | 79% |
    | 3-May | 20% |
    | 14-Jun | 1% |
    | 5-Jul | 35% |
    | 15-Jul | 81% |
    | 2-Aug | 56% |
    | 9-Aug | 17% |

Summary:
    The chart illustrates the percentage of S&P 500 market capitalization subject to buyback blackouts over a period from mid-March to early August. The Y-axis represents the percentage from 0% to 100%, while the X-axis tracks time in weekly intervals. The data shows two distinct cycles of increasing and decreasing blackout periods. The first cycle peaked around April 19th at approximately 79%, then rapidly declined to a trough of about 1% by mid-June. A second, even sharper ascent began in late June, reaching a peak of around 81% by mid-July (highlighted by the vertical line), before experiencing another rapid decline into August, settling around 17%.

Explanation:
    This chart likely reflects the cyclical nature of corporate buyback blackouts, which typically occur around earnings reporting seasons. The high percentages in April and mid-July suggest periods where a significant portion of S&P 500 companies were restricted from repurchasing their shares, potentially reducing a key source of demand in the market. Conversely, the low percentages in June indicate windows when buyback activity could resume, potentially providing support to market prices.

| Half-Month Period | Average Daily S&P 500 Return |
    |--------------------|------------------------------|
    | 1H Jan             | +0.01%                       |
    | 1H Feb             | +0.13%                       |
    | 2H Feb             | -0.14%                       |
    | 2H Mar             | +0.14%                       |
    | 1H Apr             | +0.07%                       |
    | 1H Jul             | +0.17%                       |
    | 2H Jul             | +0.07%                       |
    | 2H Aug             | -0.03%                       |
    | 2H Sep             | -0.12%                       |
    | 1H Nov             | +0.10%                       |

Summary: This bar chart displays the Average Daily S&P 500 Return for each half-month period, based on data since 2005. The X-axis spans from 1H Jan to 2H Dec, representing 24 distinct half-month periods throughout the year, while the Y-axis shows the average daily return ranging from -0.20% to +0.20%. Key trends reveal a strong seasonality in S&P 500 performance, with notable positive averages in 1H Feb, 2H Mar, 1H Jul, and 2H Nov. Conversely, significant average negative returns are observed in 2H Feb, 2H Jun, 2H Aug, and 2H Sep, with 1H Jul showing the highest positive average return and 2H Feb and 2H Sep showing the most significant average losses. The red box highlights the mixed performance typically seen from 1H July through 1H September, transitioning from strong gains to negative averages.

Explanation: This chart provides historical context for the S&P 500's seasonal performance, which can inform investment strategies or expectations for market behavior during different times of the year. The explicit title "Average Daily S&P 500 Return by Half Month Since 2005" directly ties into the broader discussion on the page about market positioning and hedge fund activity. The seasonal patterns highlight periods where the market has historically provided positive or negative returns, influencing how participants, including hedge funds, might manage their nets and overall exposure, potentially providing "cushions" or facing challenges based on these recurring trends.

page_end

Short Term:                               Weighted                          Net vs. S&P 
                  Also 
gamma    support. on                                                                                                                                                                                                                            Source:                                                                4,500      5,000     5,500    6,000     6,500     S&P 
 of                the                                                    30%      35%       -20%     -15% -10% -5% 0%                       5% 10%            15%                                                                                     Jan-20         2,000     2,500     3,000    3,500      4,000                                                      500
|


output:
json
{
  "entities": [
    {
      "name": "Retail Net Flow – Jul‑02‑25 Surge",
      "type": "Metric",
      "super_class": "Metric",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Weekly retail equity inflow spike",
      "description": "Current‑week retail net flow hit USD 3.9 bn—the 2025 year‑to‑date high—far above historical norm of USD 1.85 bn.",
      "properties": {
        "date": "2025‑07‑02",
        "value_usd_bn": 3.9,
        "historical_avg_usd_bn": 1.85
      }
    },
    {
      "name": "Retail Net Flow – Aug‑22‑25 Lull",
      "type": "Metric",
      "super_class": "Metric",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Weekly retail equity inflow low",
      "description": "Current‑week retail flow slipped to USD 0.30 bn, below the historical seasonal level of USD 0.50 bn.",
      "properties": {
        "date": "2025‑08‑22",
        "value_usd_bn": 0.3,
        "historical_avg_usd_bn": 0.50
      }
    },
    {
      "name": "Retail Demand Trend Jun–Aug 2025",
      "type": "Factor",
      "super_class": "Factor",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Trajectory of retail equity flows",
      "description": "Flows swung from peak USD 3.9 bn in early July to sub‑historical levels by mid‑August, signalling waning retail appetite.",
      "properties": {
        "peak_usd_bn": 3.9,
        "latest_usd_bn": 0.3
      }
    },
    {
      "name": "S&P 500 Buyback Blackout – Jul‑15‑25 Peak",
      "type": "Event",
      "super_class": "Event",
      "custom_type": null,
      "type_explanation": null,
      "brief": "81 % of market cap in blackout",
      "description": "Corporate repurchase blackout covered 81 % of S&P market cap during earnings quiet period.",
      "properties": {
        "date": "2025‑07‑15",
        "blackout_pct_mkt_cap": 81
      }
    },
    {
      "name": "S&P 500 Buyback Blackout Cycle 2025",
      "type": "Factor",
      "super_class": "Factor",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Cyclical restriction on buybacks",
      "description": "Buyback blackout ranged 1 %–81 % of cap between Mar‑Aug; twin peaks Apr‑19 (79 %) and Jul‑15 (81 %).",
      "properties": {}
    },
    {
      "name": "Seasonal Positive Window 1H Jul",
      "type": "Metric",
      "super_class": "Metric",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Avg daily S&P return +0.17 %",
      "description": "Since 2005, first half of July shows strongest positive seasonality.",
      "properties": {
        "avg_daily_return_pct": 0.17
      }
    },
    {
      "name": "Seasonal Negative Window 2H Sep",
      "type": "Metric",
      "super_class": "Metric",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Avg daily S&P return –0.12 %",
      "description": "Second half of September historically delivers one of the worst average returns.",
      "properties": {
        "avg_daily_return_pct": -0.12
      }
    },
    {
      "name": "Market Fragility Risk Summer 2025",
      "type": "Factor",
      "super_class": "Factor",
      "custom_type": null,
      "type_explanation": null,
      "brief": "Heightened vulnerability to pull‑backs",
      "description": "Slowing retail and systematic inflows plus buyback blackout raise correction risk into late summer.",
      "properties": {}
    }
  ],
  "relationships": [
    {
      "source": "Retail Net Flow – Jul‑02‑25 Surge",
      "target": "Retail Demand Trend Jun–Aug 2025",
      "rel_type": "Other",
      "custom_rel_type": "PEAK_OF",
      "rel_explanation": "Defines the high point",
      "properties": {
        "confidence": 95
      },
      "description": "Early‑July spike marks trend peak"
    },
    {
      "source": "Retail Net Flow – Aug‑22‑25 Lull",
      "target": "Retail Demand Trend Jun–Aug 2025",
      "rel_type": "Other",
      "custom_rel_type": "TROUGH_OF",
      "rel_explanation": "Defines the low point",
      "properties": {
        "confidence": 95
      },
      "description": "Mid‑Aug lull marks trend low"
    },
    {
      "source": "Retail Demand Trend Jun–Aug 2025",
      "target": "Market Fragility Risk Summer 2025",
      "rel_type": "CAUSES",
      "custom_rel_type": null,
      "rel_explanation": null,
      "properties": {
        "probability": 0.7,
        "lag_days": 0,
        "impact_value": null,
        "unit": null,
        "non_linear": null,
        "confidence": 80
      },
      "description": "Fading retail demand reduces support"
    },
    {
      "source": "S&P 500 Buyback Blackout – Jul‑15‑25 Peak",
      "target": "S&P 500 Buyback Blackout Cycle 2025",
      "rel_type": "Other",
      "custom_rel_type": "PEAK_OF",
      "rel_explanation": "Cycle high",
      "properties": {
        "confidence": 90
      },
      "description": "Highest blackout share in 2025 cycle"
    },
    {
      "source": "S&P 500 Buyback Blackout Cycle 2025",
      "target": "Market Fragility Risk Summer 2025",
      "rel_type": "IMPACTS",
      "custom_rel_type": null,
      "rel_explanation": null,
      "properties": {
        "probability": 0.8,
        "lag_days": 0,
        "impact_value": null,
        "unit": null,
        "non_linear": null,
        "confidence": 80
      },
      "description": "Blackout removes corporate bid"
    },
    {
      "source": "Seasonal Positive Window 1H Jul",
      "target": "Retail Net Flow – Jul‑02‑25 Surge",
      "rel_type": "Other",
      "custom_rel_type": "COINCIDES_WITH",
      "rel_explanation": "Seasonal tailwind",
      "properties": {
        "confidence": 70
      },
      "description": "Seasonality aligns with retail buying"
    },
    {
      "source": "Seasonal Negative Window 2H Sep",
      "target": "Market Fragility Risk Summer 2025",
      "rel_type": "Other",
      "custom_rel_type": "FORWARDS_RISK",
      "rel_explanation": "Upcoming weak season",
      "properties": {
        "confidence": 65
      },
      "description": "Historically weak period approaches"
    }
  ],
}

</triplet_example>."""

# ---------------------------------------------------------------------------
# 3. LLM-BASED JUDGE METRIC
# ---------------------------------------------------------------------------

_JUDGE_CACHE: Dict[Tuple[str,str], float] = {}

_JUDGE_SYSTEM_PROMPT = ("""
    You are a world-class financial knowledge-graph evaluator.  You will receive:

  • GOLD JSON: with “entities”, “relationships”, and optional “scenarios”.  
  • PREDICTED JSON: same schema.

Do a two-step process:

**Deep Qualitative Analysis**  
   • **Accuracy & Completeness**: which entities/relations are correct, missing, or spurious?  
   • **Semantic Faithfulness**: does the prediction capture the *meaning* and nuance of the gold?  
   • **Content Richness**: are scenarios or contextual details present when expected?  
   • **Clarity & Structure**: is the JSON well-organized and unambiguous?  
   • **No Hallucinations**: does it avoid introducing unsupported facts?
  
    Given a gold (expected) JSON and a predicted JSON, return a single quality score
    On a scale 0–1 (inclusive), give a single score that reflects *all* of the above.  1.0 = perfect semantic match + full coverage; 0.0 = entirely wrong.
    wrong. Respond with ONLY the numeric score.
""")


def _call_judge_llm(gold_json: str, pred_json: str) -> float:
    """Query GPT-4o-mini judge and parse score."""
    key = (gold_json, pred_json)
    if key in _JUDGE_CACHE:
        return _JUDGE_CACHE[key]

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))

        legacy_completion_models = {
            "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"
        }

        # ------------------------------------------------------------------
        # Decide which endpoint to call:
        #  • Legacy completion models (text-davinci-003, etc.) → /v1/completions
        #  • Everything else (o3, gpt-4o-mini, ...)           → /v1/chat/completions
        # ------------------------------------------------------------------
        if JUDGE_MODEL in legacy_completion_models:
            # -------- Completion-style (legacy) --------
            prompt = (
                _JUDGE_SYSTEM_PROMPT.strip() + "\n\n" +
                "GOLD JSON:\n" + gold_json.strip() + "\n\n" +
                "PREDICTED JSON:\n" + pred_json.strip() + "\n\n" +
                "Score:"
            )
            logger.info("📝 [Judge] Invoking completion model '%s'", JUDGE_MODEL)
            resp = client.completions.create(
                model=JUDGE_MODEL,
                prompt=prompt,
                temperature=0.0,
                max_tokens=4,
            )
            text = resp.choices[0].text.strip()
            logger.info("📊 [Judge] Raw completion response: %s", text)
        else:
            # -------- Chat-style model --------
            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "GOLD JSON:\n" + gold_json.strip() + "\n\n" +
                        "PREDICTED JSON:\n" + pred_json.strip() + "\n\n" +
                        "Score:"
                    ),
                },
            ]
            try:
                logger.info("📝 [Judge] Invoking chat model '%s'", JUDGE_MODEL)
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                   
                )
                text = resp.choices[0].message.content.strip()
                logger.info("📊 [Judge] Raw chat response: %s", text)
            except Exception as chat_err:
                # Fallback: some very old completion-only models might mis-detect
                logger.warning("⚠️  [Judge] Chat endpoint failed (%s); retrying via completions", chat_err)
                resp = client.completions.create(
                    model=JUDGE_MODEL,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=4,
                )
                text = resp.choices[0].text.strip()
                logger.info("📊 [Judge] Raw fallback completion response: %s", text)
        # Extract first float in response
        import re
        match = re.search(r"[01](?:\.\d+)?", text)  # robust regex for 0–1 range
        score = float(match.group(0)) if match else 0.0
        logger.info("✅ [Judge] Parsed score: %.3f", score)
    except Exception as e:
        logger.error("Judge LLM error: %s", e)
        breakpoint()
        score = 0.0

    _JUDGE_CACHE[key] = score
    return score


def llm_judge_metric(example, prediction, trace=None) -> float:
    """Metric wrapper for DSPy using LLM judge."""
    try:
        if isinstance(prediction, str):
            pred_json = prediction.strip()
        else:
            pred_json = json.dumps(prediction, separators=(",", ":"))

        if not pred_json:
            return 0.0

        gold_json = getattr(example, "expected_triplets", "").strip()
        if not gold_json:
            return 0.0

        # Invoke LLM judge and obtain a quality score in [0,1]
        score = _call_judge_llm(gold_json, pred_json)

        if trace and hasattr(trace, "__dict__"):
            trace.score = score
        return score
    except Exception as e:
        logger.error("Metric wrapper error: %s", e)
        return 0.0

# ---------------------------------------------------------------------------
# 4. EXTRACTOR MODULE
# ---------------------------------------------------------------------------

class TripletExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(TripletExtractionSignature)

    def forward(self, text_chunk: str):  # -> Dict[str,Any]
        try:
            result = self.extract(text_chunk=text_chunk)
            return json.loads(result.triplets_json)
        except Exception as e:
            logger.error("Extraction error: %s", e)
            return {"entities": [], "relationships": []}

# ---------------------------------------------------------------------------
# 5. PIPELINE CLASS
# ---------------------------------------------------------------------------

class LLMJudgeMIPROPipeline:
    def __init__(self, few_shots_dir: str = "few_shots", out_dir: str = "mipro_llm_judge_results"):
        self.few_shots_dir = few_shots_dir
        self.out_dir = out_dir
        self.extractor = TripletExtractor()
        self.optimized_extractor = None
        print(f"📂 INPUT: {os.path.abspath(self.few_shots_dir)}")
        print(f"📁 OUTPUT: {os.path.abspath(self.out_dir)}")

    # ------ Data loading (same as previous) ------
    def load_examples(self) -> List[dspy.Example]:
        examples: List[dspy.Example] = []
        i = 2
        while True:
            example_dir = os.path.join(self.few_shots_dir, f"few_shot_{i}")
            if not os.path.exists(example_dir):
                break
            try:
                with open(os.path.join(example_dir, "chunk.txt"), "r", encoding="utf-8") as f:
                    text_chunk = f.read().strip()
                with open(os.path.join(example_dir, "triplets.json"), "r", encoding="utf-8") as f:
                    expected_triplets = json.dumps(json.load(f))
                ex = dspy.Example(text_chunk=text_chunk, expected_triplets=expected_triplets).with_inputs("text_chunk")
                examples.append(ex)
                logger.info("Loaded %s", example_dir)
            except Exception as e:
                logger.error("Failed to load %s: %s", example_dir, e)
            i += 1
        logger.info("Total examples: %d", len(examples))
        return examples

    # ------ Run pipeline ------
    def run(self):
        logger.info("🚀 LLM-Judge MIPRO Pipeline starting ...")
        configure_llm()
        examples = self.load_examples()
        if len(examples) < 2:
            logger.error("Need at least 2 examples")
            return

        # split: last 5 (max) for validation
        num_val = min(5, len(examples)//2)
        train_examples = examples
        val_examples = examples[-num_val:]
        logger.info("Train %d  |  Val %d", len(train_examples), len(val_examples))

        # ------------------------------------------------------------------
        # MIPRO settings tuned to encourage few-shot prompting
        #  • auto="aggressive"  → broader search
        #  • max_bootstrapped_demos increased
        # ------------------------------------------------------------------
        mipro = MIPROv2(
            metric=llm_judge_metric,
            auto="medium",
            max_bootstrapped_demos=4,    # search more demo sets      # allow bigger demo sets
            teacher_settings={"instructions": [TripletExtractionSignature.instructions]},
        )

        self.optimized_extractor = mipro.compile(
            self.extractor,
            trainset=train_examples,
            max_bootstrapped_demos=3,
            requires_permission_to_run=False,
        )
        logger.info("✅ Optimization done")

        # Evaluate
        evaluate = dspy.Evaluate(devset=val_examples, metric=llm_judge_metric, display_progress=True)
        score = evaluate(self.optimized_extractor)
        logger.info("Final score: %.3f", score)

        self._save_results(score, val_examples)

        # 7. Persist the entire optimized program for later reuse
        self._save_program()

        # 8. Save the demos / few-shot examples actually chosen by MIPRO
        self._save_demos()

    # ------ Save ------
    def _save_results(self, score: float, val_examples):
        os.makedirs(self.out_dir, exist_ok=True)
        meta = {
            "method": "MIPRO-LLM-Judge",
            "timestamp": datetime.now().isoformat(),
            "model": EXTRACT_MODEL_ALIAS,
            "judge_model": JUDGE_MODEL,
            "score": float(score),
            "train_examples": len(val_examples),
        }
        with open(os.path.join(self.out_dir, "results.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Results saved → %s", self.out_dir)

    def _save_program(self):
        """Persist the optimized DSPy program (architecture + state) for reuse."""
        try:
            prog_dir = os.path.join(self.out_dir, "optimized_program")
            os.makedirs(prog_dir, exist_ok=True)
            self.optimized_extractor.save(prog_dir, save_program=True)
            logger.info("📦 Saved optimized program → %s", prog_dir)
        except Exception as e:
            logger.warning("Could not save optimized program: %s", e)

    def _save_demos(self):
        """Write the final few-shot demonstrations to a human-readable file."""
        try:
            demos_out = os.path.join(self.out_dir, "best_demos.json")
            import json as _json

            # DSPy stores demos as list[dspy.Example] or list[dict]
            demos_list = []

            # Helper to add demos safely
            def _append_demos(ds):
                for d in ds:
                    demos_list.append(d.toDict() if hasattr(d, "toDict") else d)

            # Top-level demos
            _append_demos(getattr(self.optimized_extractor, "demos", []))

            # Crawl predictors (1-level) for embedded demos
            for pred in getattr(self.optimized_extractor, "predictors", lambda: [])():
                _append_demos(getattr(pred, "demos", []))

            # Deduplicate (based on stringified representation)
            demos_list_unique = list({json.dumps(d, sort_keys=True) for d in demos_list})
            demos_list = [json.loads(s) for s in demos_list_unique]

            with open(demos_out, "w") as f:
                _json.dump(demos_list, f, indent=2)

            logger.info("📄 Saved %d best demos → %s", len(demos_list), demos_out)
        except Exception as e:
            logger.warning("Could not save demos: %s", e)

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    INPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/few_shots"
    OUTPUT_DIR = "/Users/aaryangoyal/Desktop/Aaryan_folder/mipro_llm_judge_results"
    pipe = LLMJudgeMIPROPipeline(INPUT_DIR, OUTPUT_DIR)
    pipe.run()


if __name__ == "__main__":
    main() 