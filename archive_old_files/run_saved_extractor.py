#!/usr/bin/env python3
"""
run_saved_extractor.py
======================

Utility to load a previously-saved DSPy optimized program (saved with
mipro_llm_judge_pipeline.py) and run it on one text chunk or a whole
folder of chunks.

Usage
-----
1. Single file → JSON next to it:
   python run_saved_extractor.py --program saved/optimized_program \
                                 --input  my_chunk.txt

2. Directory of *.txt files → outputs to OUT_DIR (mirrors structure):
   python run_saved_extractor.py --program saved/optimized_program \
                                 --input  test_chunks/             \
                                 --out    outputs/

3. Direct text input → JSON in OUT_DIR:
   python run_saved_extractor.py --program saved/optimized_program \
                                 --chunk_text "Your text chunk here" \
                                 --out    outputs/

Notes
~~~~~
• Expects text chunks in plain UTF-8.
• Output JSON file gets the same base name with suffix `.triplets.json`.
• Requires the same LLM as the saved program, here we use gpt-4o-mini.
"""

import argparse, os, pathlib, json, sys
import dspy

# ----- OPENAI KEY -------------------------------------------------
OPENAI_API_KEY = ""          #  ←  paste your real key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ------------------------------------------------------------------

# ────────────────────────── Parse arguments ──────────────────────────
parser = argparse.ArgumentParser(
    description="Run a DSPy optimized program on text chunks."
)
parser.add_argument(
    "--program",
    type=pathlib.Path,
    required=True,
    help="Path to the saved DSPy optimized program directory.",
)
parser.add_argument(
    "--input",
    type=pathlib.Path,
    help="Path to a single chunk.txt file or a directory containing multiple *.txt files."
)
parser.add_argument(
    "--out",
    type=pathlib.Path,
    default=pathlib.Path("./extraction_outputs"),
    help="Directory where JSON outputs will be written.",
)
parser.add_argument(
    "--chunk_text",
    type=str,
    help="Direct text chunk to process, instead of reading from a file.",
)

args = parser.parse_args()

prog_dir = args.program
input_path = args.input
out_root = args.out
chunk_text_input = args.chunk_text

if not prog_dir.exists():
    sys.exit(f"❌ Program dir {prog_dir} not found")

if chunk_text_input is None and input_path is None:
    sys.exit("❌ Either --input or --chunk_text must be provided.")
if chunk_text_input is not None and input_path is not None:
    sys.exit("❌ Cannot use both --input and --chunk_text simultaneously.")
if input_path is not None and not input_path.exists():
    sys.exit(f"❌ Input path {input_path} not found")

out_root.mkdir(parents=True, exist_ok=True)

# ────────────────── Configure LLM & load program ──────────────────
print("⚙️  Configuring LLM …")
dspy.settings.configure(
    lm=dspy.LM(
        "openai/gpt-4o-mini", temperature=1.0, max_tokens=20000
    )
)
print(f"🔄 Loading program from {prog_dir} …")
extractor = dspy.load(str(prog_dir))
print("✅ Program loaded")

# ────────────────── Helper to process one file ──────────────────

def process_file(txt_path: pathlib.Path):
    chunk = txt_path.read_text().strip()
    if not chunk:
        print(f"⚠️  Skipping empty file {txt_path}")
        return

    result = extractor(text_chunk=chunk)
    json_str = result.triplets_json if hasattr(result, "triplets_json") else result
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
    except Exception:
        data = json_str  # fallback raw

    rel_path = txt_path.relative_to(input_path.parent if input_path.is_file() else input_path)
    out_file = out_root / rel_path.with_suffix(".triplets.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Pydantic object to dictionary before dumping to JSON
    data_to_save = data.model_dump() if hasattr(data, 'model_dump') else data
    out_file.write_text(json.dumps(data_to_save, indent=2))
    print(f"💾 {txt_path} → {out_file}")

# ────────────────── Run extraction ──────────────────
if chunk_text_input:
    # Process the direct chunk text input
    print("Processing direct chunk text input...")
    result = extractor(text_chunk=chunk_text_input)
    json_str = result.triplets_json if hasattr(result, "triplets_json") else result
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
    except Exception:
        data = json_str  # fallback raw

    # Determine output file name for direct chunk text
    output_filename = "direct_chunk_output.triplets.json"
    out_file = out_root / output_filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = data.model_dump() if hasattr(data, 'model_dump') else data
    out_file.write_text(json.dumps(data_to_save, indent=2))
    print(f"💾 Direct chunk text → {out_file}")

elif input_path.is_file():
    process_file(input_path)
else: # input_path is a directory
    for sub_dir in sorted(input_path.iterdir()):
        if sub_dir.is_dir() and sub_dir.name.startswith("few_shot_"):
            try:
                shot_num = int(sub_dir.name.split("_")[2])
                if shot_num >= 102:
                    for txt in sub_dir.rglob("*.txt"):
                        process_file(txt)
            except (ValueError, IndexError):
                print(f"⚠️  Skipping non-standard few-shot directory: {sub_dir.name}")

print("🎉 Extraction complete") 