#!/usr/bin/env python3
"""
Production Model Runner
======================

Utility to load a previously-saved DSPy optimized program and run it on 
text chunks for production inference.

Usage:
------
1. Single file → JSON next to it:
   python model_runner.py --program saved/optimized_program --input my_chunk.txt

2. Directory of *.txt files → outputs to OUT_DIR:
   python model_runner.py --program saved/optimized_program --input test_chunks/ --out outputs/

3. Direct text input → JSON in OUT_DIR:
   python model_runner.py --program saved/optimized_program --chunk_text "Your text" --out outputs/
"""

import argparse, os, pathlib, json, sys
import dspy

from src.config import configure_llm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a DSPy optimized program on text chunks.")
    
    parser.add_argument(
        "--program", type=pathlib.Path, required=True,
        help="Path to the saved DSPy optimized program directory."
    )
    parser.add_argument(
        "--input", type=pathlib.Path,
        help="Path to a single chunk.txt file or directory containing *.txt files."
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("./extraction_outputs"),
        help="Directory where JSON outputs will be written."
    )
    parser.add_argument(
        "--chunk_text", type=str,
        help="Direct text chunk to process instead of reading from a file."
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model to use for inference (default: gpt-4o-mini)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if not args.program.exists():
        sys.exit(f"❌ Program directory {args.program} not found")

    if args.chunk_text is None and args.input is None:
        sys.exit("❌ Either --input or --chunk_text must be provided.")
    
    if args.chunk_text is not None and args.input is not None:
        sys.exit("❌ Cannot use both --input and --chunk_text simultaneously.")
    
    if args.input is not None and not args.input.exists():
        sys.exit(f"❌ Input path {args.input} not found")

    args.out.mkdir(parents=True, exist_ok=True)


def load_extractor(program_dir: pathlib.Path, model: str):
    """Load the saved DSPy program."""
    print("⚙️  Configuring LLM...")
    configure_llm(model=model)
    
    print(f"🔄 Loading program from {program_dir}...")
    extractor = dspy.load(str(program_dir))
    print("✅ Program loaded")
    
    return extractor


def process_file(txt_path: pathlib.Path, extractor, output_root: pathlib.Path, input_base: pathlib.Path):
    """Process a single text file."""
    chunk = txt_path.read_text().strip()
    if not chunk:
        print(f"⚠️  Skipping empty file {txt_path}")
        return

    # Run extraction
    result = extractor(text_chunk=chunk)
    
    # Handle different result formats
    if hasattr(result, "model_dump"):
        # Pydantic model
        data = result.model_dump()
    elif hasattr(result, "triplets_json"):
        # DSPy result with JSON string
        try:
            data = json.loads(result.triplets_json)
        except json.JSONDecodeError:
            data = {"raw_output": result.triplets_json}
    else:
        # Direct dictionary or other format
        data = result

    # Determine output path
    rel_path = txt_path.relative_to(input_base)
    out_file = output_root / rel_path.with_suffix(".triplets.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save result
    out_file.write_text(json.dumps(data, indent=2))
    print(f"💾 {txt_path} → {out_file}")


def process_direct_text(text: str, extractor, output_dir: pathlib.Path):
    """Process direct text input."""
    print("Processing direct text input...")
    
    result = extractor(text_chunk=text)
    
    # Handle different result formats
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "triplets_json"):
        try:
            data = json.loads(result.triplets_json)
        except json.JSONDecodeError:
            data = {"raw_output": result.triplets_json}
    else:
        data = result

    # Save to output directory
    out_file = output_dir / "direct_input_result.triplets.json"
    out_file.write_text(json.dumps(data, indent=2))
    print(f"💾 Direct text input → {out_file}")


def process_directory(input_dir: pathlib.Path, extractor, output_dir: pathlib.Path):
    """Process all .txt files in a directory."""
    txt_files = list(input_dir.rglob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  No .txt files found in {input_dir}")
        return
    
    print(f"📁 Processing {len(txt_files)} files...")
    
    for txt_file in txt_files:
        try:
            process_file(txt_file, extractor, output_dir, input_dir)
        except Exception as e:
            print(f"❌ Error processing {txt_file}: {e}")


def main():
    """Main execution function."""
    args = parse_arguments()
    validate_arguments(args)
    
    # Load the saved extractor
    try:
        extractor = load_extractor(args.program, args.model)
    except Exception as e:
        sys.exit(f"❌ Failed to load extractor: {e}")
    
    # Process input
    if args.chunk_text:
        process_direct_text(args.chunk_text, extractor, args.out)
    elif args.input.is_file():
        process_file(args.input, extractor, args.out, args.input.parent)
    else:  # Directory
        process_directory(args.input, extractor, args.out)
    
    print("🎉 Extraction complete!")


if __name__ == "__main__":
    main()