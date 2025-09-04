#!/usr/bin/env python3
"""
Batch Text Processor for Financial Triplet Extraction
======================================================
CLI utility that:
1. Takes text chunks from various sources (stdin, files, directories).
2. Sends them to OpenAI models for triplet extraction.
3. Saves both chunks and JSON results.
4. Supports concurrent processing and deduplication.
"""

import os, json, re, hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import openai
import argparse

from src.config import configure_openai_direct

# Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_FEW_SHOTS_DIR = Path("./few_shots")
PAGE_SPLIT_RE = re.compile(r"page_end", re.IGNORECASE)

# Thread-safe directory creation
_dir_lock = threading.Lock()


def next_example_dir(root: Path) -> Path:
    """Thread-safe creation of the next auto-incremented few_shot directory."""
    with _dir_lock:
        root.mkdir(exist_ok=True)
        existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("few_shot_")]
        nums = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
        n = max(nums) + 1 if nums else 1
        d = root / f"few_shot_{n}"
        d.mkdir()
        return d


def query_model(chunk: str, model: str, prompt_template: str) -> str:
    """Query OpenAI model for triplet extraction."""
    
    system_msg = {
        "role": "system",
        "content": "You are an assistant that extracts financial triplet JSON strictly following the instructions provided."
    }
    user_msg = {
        "role": "user",
        "content": prompt_template.replace("<content>", chunk)
    }

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[system_msg, user_msg],
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def process_chunk(chunk: str, output_dir: Path, model: str, prompt_template: str) -> Path:
    """Process a single chunk and save outputs."""
    json_text = query_model(chunk, model, prompt_template)

    # Validate/prettify JSON
    try:
        data = json.loads(json_text)
        json_text_pretty = json.dumps(data, indent=2)
    except Exception:
        json_text_pretty = json_text  # Save raw if invalid JSON

    dest = next_example_dir(output_dir)
    (dest / "chunk.txt").write_text(chunk)
    (dest / "triplets.json").write_text(json_text_pretty)
    
    meta = {
        "created": datetime.now().isoformat(),
        "model": model
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    return dest


def load_chunks_from_files(input_paths: list[Path]) -> list[str]:
    """Load text chunks from files, splitting on page markers."""
    chunks = []
    
    for path in input_paths:
        if path.is_file() and path.suffix == '.txt':
            text = path.read_text("utf-8", errors="ignore")
            parts = [p.strip() for p in PAGE_SPLIT_RE.split(text) if p.strip()]
            chunks.extend(parts)
        elif path.is_dir():
            for txt_path in path.rglob("*.txt"):
                text = txt_path.read_text("utf-8", errors="ignore")
                parts = [p.strip() for p in PAGE_SPLIT_RE.split(text) if p.strip()]
                chunks.extend(parts)
    
    return chunks


def deduplicate_chunks(chunks: list[str], existing_dir: Path) -> list[str]:
    """Remove chunks that already exist in the output directory."""
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    existing_hashes = set()
    if existing_dir.exists():
        for d in existing_dir.iterdir():
            if d.is_dir():
                chunk_file = d / "chunk.txt"
                if chunk_file.exists():
                    try:
                        existing_hashes.add(_hash(chunk_file.read_text("utf-8", errors="ignore")))
                    except Exception:
                        pass

    orig_len = len(chunks)
    chunks = [c for c in chunks if _hash(c) not in existing_hashes]
    if orig_len != len(chunks):
        print(f"↪️  Skipping {orig_len - len(chunks)} chunks already processed")

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Batch process text chunks for triplet extraction")
    parser.add_argument("--input", "-i", type=Path, nargs="+", help="Input files or directories")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_FEW_SHOTS_DIR, 
                       help="Output directory for results")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--prompt", "-p", type=Path, help="Custom prompt template file")
    
    args = parser.parse_args()

    # Configure OpenAI
    try:
        configure_openai_direct(model=args.model)
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    # Load prompt template
    if args.prompt and args.prompt.exists():
        prompt_template = args.prompt.read_text()
    else:
        # Use default simple prompt
        prompt_template = """
Extract entities and relationships from the following financial text as JSON:

<content>

Return valid JSON with "entities" and "relationships" keys only.
"""

    # Load and process chunks
    if args.input:
        chunks = load_chunks_from_files(args.input)
        print(f"🔍 Loaded {len(chunks)} chunks from input files")
    else:
        print("❌ No input files specified")
        return

    if not chunks:
        print("⚠️  No chunks to process")
        return

    # Deduplicate
    chunks = deduplicate_chunks(chunks, args.output)
    
    if not chunks:
        print("⚠️  No new chunks to process")
        return

    print(f"\n⏳ Processing {len(chunks)} chunk(s) with {args.workers} workers...")

    # Process in parallel
    with ThreadPoolExecutor(max_workers=min(args.workers, len(chunks))) as exe:
        future_to_chunk = {
            exe.submit(process_chunk, c, args.output, args.model, prompt_template): c 
            for c in chunks
        }
        
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                dest_path = future.result()
                print(f"✅ Processed chunk (first 40 chars: {chunk[:40]!r}...) → {dest_path}")
            except Exception as e:
                print(f"❌ Error processing chunk: {e}")

    print(f"\n🎉 Batch processing completed! Results saved to {args.output}/")


if __name__ == "__main__":
    main()