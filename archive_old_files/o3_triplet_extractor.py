#!/usr/bin/env python3
"""
O3 Triplet Extractor
====================
CLI utility that:
1. Takes a text chunk (stdin or --file).
2. Sends it to OpenAI gpt-4o-mini ("o3" style) with the triplet-extraction prompt.
3. Receives JSON triplets.
4. Saves both chunk and JSON into a new folder few_shots/few_shot_N/ (auto-increment).
"""

import os, json, re, hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import openai

# ---------------------------------------------------------------------------
# INPUT MODE 1: Manual CHUNKS list  (leave empty to disable)
# INPUT MODE 2: DIRECTORY of .txt files. Each chunk is text between successive
#               'page_end' markers (case-insensitive) inside each file.
#               Set INPUT_DIR to the folder path. If both modes are empty, script exits.
# ---------------------------------------------------------------------------

CHUNKS: list[str] = []  # Fill to use manual mode; otherwise leave []

# Directory containing .txt files. Example:
INPUT_DIR = Path("/Users/aaryangoyal/Desktop/Aaryan_folder/graph_structures/pdf_parser/data/converted_outputs")

# Regular expression to split pages. Adjust if marker differs.
PAGE_SPLIT_RE = re.compile(r"page_end", re.IGNORECASE)

# ---------- CONFIG ----------
OPENAI_API_KEY = ""  # replace or set via env
MODEL = "o3"  # O3 model shortcut
FEW_SHOTS_DIR = Path("/Users/aaryangoyal/Desktop/Aaryan_folder/few_shots")

PROMPT_TEMPLATE = Path("/Users/aaryangoyal/Desktop/Aaryan_folder/triplet_example.md").read_text()

# ---------- helper ----------

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


def query_o3(chunk: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    system_msg = {
        "role": "system",
        "content": "You are an assistant that extracts financial triplet JSON strictly following the instructions provided."
    }
    user_msg = {
        "role": "user",
        "content": PROMPT_TEMPLATE.replace("<content>", chunk)
    }

    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[system_msg, user_msg],
        
    )
    return resp.choices[0].message["content"].strip()


def _process_chunk(chunk: str) -> Path:
    """Query the model for *chunk* and save outputs. Returns the destination path."""
    json_text = query_o3(chunk)

    # validate / prettify JSON
    try:
        data = json.loads(json_text)
        json_text_pretty = json.dumps(data, indent=2)
    except Exception:
        json_text_pretty = json_text  # save raw

    dest = next_example_dir(FEW_SHOTS_DIR)
    (dest / "chunk.txt").write_text(chunk)
    (dest / "triplets.json").write_text(json_text_pretty)
    meta = {
        "created": datetime.now().isoformat(),
        "model": MODEL
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    return dest

# ---------- new helper to update existing few_shots ----------

def update_first_n_dirs(n: int = 48):
    """Re-run extraction for the first *n* few_shot directories, overwriting triplets.json."""
    dirs = sorted(
        [d for d in FEW_SHOTS_DIR.iterdir() if d.is_dir() and d.name.startswith("few_shot_")],
        key=lambda p: int(p.name.split("_")[-1])
    )
    if not dirs:
        print("⚠️  No few_shot directories found.")
        return

    targets = dirs[:n]
    print(f"🔄 Updating triplets for {len(targets)} existing few_shot directories …")

    for d in targets:
        chunk_file = d / "chunk.txt"
        if not chunk_file.exists():
            print(f"⚠️  {chunk_file} missing, skipping.")
            continue

        chunk = chunk_file.read_text("utf-8", errors="ignore").strip()
        if not chunk:
            print(f"⚠️  Empty chunk in {chunk_file}, skipping.")
            continue

        try:
            json_text = query_o3(chunk)
            try:
                data = json.loads(json_text)
                json_text_pretty = json.dumps(data, indent=2)
            except Exception:
                json_text_pretty = json_text  # save raw if not valid JSON

            # Overwrite / create triplets.json
            (d / "triplets.json").write_text(json_text_pretty)
            meta = {
                "updated": datetime.now().isoformat(),
                "model": MODEL
            }
            (d / "metadata.json").write_text(json.dumps(meta, indent=2))
            print(f"✅ Updated {d.name}")
        except Exception as e:
            print(f"❌ Error updating {d.name}: {e}")

# ---------- main ----------

def main():
    # First, update the first 48 existing few_shot directories then exit.
    update_first_n_dirs(48)
    


    chunks: list[str] = [c.strip() for c in CHUNKS if c.strip()]

    # If CHUNKS list empty, attempt directory mode
    if not chunks and INPUT_DIR and INPUT_DIR.exists():
        print(f"🔍 Scanning {INPUT_DIR} for .txt files …")
        for txt_path in INPUT_DIR.rglob("*.txt"):
            text = txt_path.read_text("utf-8", errors="ignore")
            parts = [p.strip() for p in PAGE_SPLIT_RE.split(text) if p.strip()]
            chunks.extend(parts)
        print(f"📄 Collected {len(chunks)} chunks from {INPUT_DIR}")

    # ---- Deduplicate against already-saved few_shots ----
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    existing_hashes = set()
    if FEW_SHOTS_DIR.exists():
        for d in FEW_SHOTS_DIR.iterdir():
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
        print(f"↪️  Skipping {orig_len - len(chunks)} chunks already in few_shots")

    if not chunks:
        print("⚠️  No new chunks to process.")
        return

    print(f"\n⏳ Processing {len(chunks)} chunk(s) in parallel...")

    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as exe:  # up to 4 parallel requests
        future_to_chunk = {exe.submit(_process_chunk, c): c for c in chunks}
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                dest_path = future.result()
                print(f"✅ Saved result for chunk (first 40 chars: {chunk[:40]!r}...) → {dest_path}")
            except Exception as e:
                print(f"❌ Error processing chunk (first 40 chars: {chunk[:40]!r}...): {e}")


if __name__ == "__main__":
    main() 