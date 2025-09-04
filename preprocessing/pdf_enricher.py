#!/usr/bin/env python3
"""
PDF Enrichment Pipeline with Visual Analysis
============================================

End-to-end pipeline for processing financial PDFs into enriched text:
1. High-quality text extraction using LLMWhisperer
2. Figure/chart detection using MinerU layout detection
3. Visual analysis using Gemini Vision API with financial domain expertise
4. Merging text with structured figure descriptions

This creates enriched markdown text that can be processed by DSPy extractors.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Optional, TYPE_CHECKING

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    class _LoggerShim:
        def __getattr__(self, name):
            return getattr(logging, name.upper(), logging.info)
    
    logger = _LoggerShim()  # type: ignore

if TYPE_CHECKING:
    from PIL import Image

# ========================= Configuration =========================

# Financial chart analysis prompt template
FINANCIAL_CHART_PROMPT = """
SYSTEM:
You are ChartGPT – a financial analyst and visual chart interpreter.
Your job is to convert financial or economic figures into structured data and rich, narrative insights.
Ignore logos, icons, or decorative shapes. Focus only on visual elements with structured data (e.g., time series, bar/line plots, labeled tables, annotated charts).

USER:
For the attached figure **{image_filename_placeholder}**, and using the **full-page context provided**, return:

1. A detailed **Markdown table** showing the key structured data from the figure.
   - Use meaningful column headers (e.g., Date, Index, Exposure, Weekly Change, Region, etc.)
   - Do **not limit to 3 columns**. Include all visible columns in the chart.
   - Use **up to 10 rows**, rounded and abbreviated as needed (e.g., +3.1%, $2.2bn, ‑12 bps)

2. A **rich summary paragraph (4–6 lines)** that explains:
   - The key dimensions of the chart (e.g., X/Y axes, categories/legends, time range)
   - The primary trends or shifts (e.g., rising CTA exposure, regional divergence)
   - Any inflection points, peaks, or unusual values

3. A **second paragraph (2–3 lines)** giving the broader context or implication of what's shown, based on the chart and the surrounding page text.

Do **not** include entity-relationship lists or any triple backtick code blocks.

Use the format below exactly (no preamble):

| Column1 | Column2 | Column3 | ... |
|---------|---------|---------|-----|
| …       | …       | …       | …   |
...

Summary: <clear, rich multi-line description of the chart>

Explanation: <broader context or takeaway insight>
"""

PAGE_DELIMITER = "page_end"
INDENT = "    "

# Default configuration
DEFAULT_DPI = 300
DEFAULT_DEVICE = "cpu"
DEFAULT_TIMEOUT = 300
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# ========================= Data Structures =========================

class ProcessingStats(NamedTuple):
    """Statistics from processing operation."""
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    processing_time: float
    figures_processed: int

class ProcessingConfig(NamedTuple):
    """Configuration for PDF processing."""
    dpi: int = DEFAULT_DPI
    device: str = DEFAULT_DEVICE
    timeout: int = DEFAULT_TIMEOUT
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_api_key: Optional[str] = None
    llmwhisperer_api_key: Optional[str] = None
    mineru_path: Optional[Path] = None

# ========================= Utility Functions =========================

def get_api_key(key_name: str, config_value: Optional[str] = None) -> str:
    """Get API key from config or environment with proper error handling."""
    key = config_value or os.getenv(key_name)
    if not key:
        raise ValueError(f"{key_name} not provided in config or environment variables")
    return key

def add_mineru_to_path(mineru_path: Path) -> None:
    """Add MinerU to Python path for imports."""
    if mineru_path and mineru_path.exists():
        sys.path.insert(0, str(mineru_path))
    else:
        # Try to import MinerU directly first
        try:
            import mineru
            return
        except ImportError:
            raise RuntimeError(
                f"MinerU not found. Either install with 'pip install mineru' or "
                f"provide path to MinerU directory. Path tried: {mineru_path}"
            )

def load_layout_model(device: str):
    """Load MinerU layout detection model."""
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
        from mineru.utils.enum_class import ModelPath
    except ImportError:
        raise RuntimeError("MinerU not available. Install with: pip install mineru")

    atom_models = AtomModelSingleton()
    weight_root = Path(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo))
    weights_path = weight_root / ModelPath.doclayout_yolo
    return atom_models.get_atom_model(
        atom_model_name="layout",
        doclayout_yolo_weights=str(weights_path),
        device=device,
    )

def is_figure_body(detection: dict) -> bool:
    """Check if detection is a figure/chart."""
    return int(detection.get("category_id", -1)) == 3

# ========================= Core Processing Functions =========================

def extract_text_with_llmwhisperer(pdf_path: Path, api_key: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Extract text from PDF using LLMWhisperer."""
    try:
        from unstract.llmwhisperer import LLMWhispererClientV2
    except ImportError:
        raise RuntimeError("LLMWhisperer not available. Install with: pip install llmwhisperer-client")
    
    client = LLMWhispererClientV2(api_key=api_key)
    result = client.whisper(
        file_path=str(pdf_path),
        wait_for_completion=True,
        wait_timeout=timeout,
        mode="high_quality",
        page_seperator=PAGE_DELIMITER,
    )
    return result.get("extraction", {}).get("result_text", "")

def detect_and_crop_figures(pdf_path: Path, layout_model, dpi: int = DEFAULT_DPI) -> List[Tuple[int, int, "Image"]]:
    """Detect and crop figures from PDF using MinerU."""
    try:
        from mineru.utils.pdf_image_tools import load_images_from_pdf
    except ImportError:
        raise RuntimeError("MinerU not available for PDF processing")
    
    pdf_bytes = pdf_path.read_bytes()
    images_list, _ = load_images_from_pdf(pdf_bytes, dpi=dpi)
    page_images = [d["img_pil"] for d in images_list]
    detections_per_page = layout_model.batch_predict(page_images, batch_size=4)
    
    crops = []
    for page_idx, (pil_image, detections) in enumerate(zip(page_images, detections_per_page), 1):
        figure_idx = 0
        for detection in detections:
            if not is_figure_body(detection):
                continue
            figure_idx += 1
            x0, y0, x1, y1 = detection["poly"][0], detection["poly"][1], detection["poly"][4], detection["poly"][5]
            cropped_image = pil_image.crop((x0, y0, x1, y1))
            crops.append((page_idx, figure_idx, cropped_image))
    
    return crops

def create_page_text_map(llm_text: str) -> Dict[int, str]:
    """Create mapping from page numbers to text content."""
    lines = llm_text.splitlines()
    page_map: Dict[int, str] = {}
    buffer: List[str] = []
    page_num = 1
    
    for line in lines:
        if line.startswith(PAGE_DELIMITER):
            page_map[page_num] = "\n".join(buffer)
            buffer = []
            page_num += 1
        else:
            buffer.append(line)
    
    if buffer:
        page_map[page_num] = "\n".join(buffer)
    
    return page_map

def get_context_window(page_texts: Dict[int, str], current_page: int, window_size: int = 3) -> str:
    """Get context from surrounding pages (default 3-page window)."""
    context_parts = []
    
    # Previous page
    if current_page - 1 in page_texts:
        context_parts.append(f"Previous page {current_page-1}:\n{page_texts[current_page-1]}\n")
    
    # Current page
    if current_page in page_texts:
        context_parts.append(f"Current page {current_page}:\n{page_texts[current_page]}\n")
    
    # Next page
    if current_page + 1 in page_texts:
        context_parts.append(f"Next page {current_page+1}:\n{page_texts[current_page+1]}\n")
    
    return "\n".join(context_parts)

def analyze_figures_with_gemini(crops: List[Tuple[int, int, "Image"]], 
                               page_texts: Dict[int, str],
                               api_key: str,
                               model: str = DEFAULT_GEMINI_MODEL) -> Tuple[Dict[int, List[str]], ProcessingStats]:
    """Analyze figures using Gemini Vision API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Google GenerativeAI not available. Install with: pip install google-generativeai")
    
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    
    summaries: Dict[int, List[str]] = {}
    prompt_tokens = response_tokens = total_tokens = 0
    start_time = time.time()
    
    for idx, (page, figure, image) in enumerate(crops, 1):
        # Create prompt with context
        prompt_base = FINANCIAL_CHART_PROMPT.replace(
            "{image_filename_placeholder}", 
            f"page_{page}_fig_{figure}.jpg"
        )
        context_window = get_context_window(page_texts, page)
        full_prompt = f"{prompt_base}\n\nContext (3-page window):\n{context_window}\n"
        
        try:
            response = gemini_model.generate_content([full_prompt, image])
            summaries.setdefault(page, []).append(response.text.strip())
            
            # Track usage statistics
            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens += getattr(usage, "prompt_token_count", 0)
                response_tokens += getattr(usage, "candidates_token_count", 0)
                total_tokens += getattr(usage, "total_token_count", 0)
                
        except Exception as e:
            logger.error(f"Error processing figure {idx}: {e}")
            summaries.setdefault(page, []).append(f"[Error analyzing figure {figure}]")
    
    processing_time = time.time() - start_time
    stats = ProcessingStats(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        total_tokens=total_tokens,
        processing_time=processing_time,
        figures_processed=len(crops)
    )
    
    return summaries, stats

def merge_text_with_summaries(llm_text: str, summaries_by_page: Dict[int, List[str]]) -> str:
    """Merge original text with figure summaries."""
    lines = llm_text.splitlines(keepends=True)
    output: List[str] = []
    buffer: List[str] = []
    page_num = 1

    def flush_page(page: int):
        if buffer:
            output.extend(buffer)
        if summaries := summaries_by_page.get(page):
            # Add figure descriptions with proper indentation
            block = "\n\n".join(textwrap.indent(summary, INDENT) for summary in summaries)
            output.append("\n" + block + "\n\n")
        buffer.clear()

    for line in lines:
        if line.startswith(PAGE_DELIMITER):
            flush_page(page_num)
            output.append(line)
            page_num += 1
        else:
            buffer.append(line)
    
    if buffer:
        flush_page(page_num)
    
    return "".join(output)

# ========================= Main Pipeline Class =========================

class PDFEnrichmentPipeline:
    """Complete PDF enrichment pipeline."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.layout_model = None
        
    def _initialize_layout_model(self):
        """Initialize layout model if needed."""
        if self.layout_model is None:
            if self.config.mineru_path:
                add_mineru_to_path(self.config.mineru_path)
            self.layout_model = load_layout_model(self.config.device)
    
    def process_pdf(self, pdf_path: Path, output_path: Optional[Path] = None) -> Tuple[str, ProcessingStats]:
        """Process a single PDF file."""
        start_time = time.time()
        
        # Initialize model
        self._initialize_layout_model()
        
        # Get API keys
        gemini_key = get_api_key("GEMINI_API_KEY", self.config.gemini_api_key)
        llmw_key = get_api_key("LLMWHISPERER_API_KEY", self.config.llmwhisperer_api_key)
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text
        logger.info("Extracting text with LLMWhisperer...")
        llm_text = extract_text_with_llmwhisperer(pdf_path, llmw_key, self.config.timeout)
        page_texts = create_page_text_map(llm_text)
        
        # Step 2: Detect and crop figures
        logger.info("Detecting and cropping figures...")
        crops = detect_and_crop_figures(pdf_path, self.layout_model, self.config.dpi)
        
        logger.info(f"Found {len(crops)} figures across {len(page_texts)} pages")
        
        # Step 3: Analyze figures with Gemini
        if crops:
            logger.info("Analyzing figures with Gemini Vision...")
            summaries, gemini_stats = analyze_figures_with_gemini(
                crops, page_texts, gemini_key, self.config.gemini_model
            )
        else:
            logger.info("No figures found, skipping vision analysis")
            summaries = {}
            gemini_stats = ProcessingStats(0, 0, 0, 0, 0)
        
        # Step 4: Merge text with summaries
        logger.info("Merging text with figure descriptions...")
        enriched_text = merge_text_with_summaries(llm_text, summaries)
        
        # Step 5: Save output
        if output_path:
            output_path.write_text(enriched_text, encoding="utf-8")
            logger.success(f"Enriched text saved to: {output_path}")
        
        # Create final stats
        total_time = time.time() - start_time
        final_stats = ProcessingStats(
            prompt_tokens=gemini_stats.prompt_tokens,
            response_tokens=gemini_stats.response_tokens, 
            total_tokens=gemini_stats.total_tokens,
            processing_time=total_time,
            figures_processed=gemini_stats.figures_processed
        )
        
        logger.success(
            f"Processing complete: {final_stats.figures_processed} figures, "
            f"{final_stats.total_tokens} tokens, {final_stats.processing_time:.1f}s"
        )
        
        return enriched_text, final_stats

# ========================= Convenience Functions =========================

def process_pdf(pdf_path: Path, 
                output_path: Optional[Path] = None,
                config: Optional[ProcessingConfig] = None) -> Tuple[str, ProcessingStats]:
    """Convenience function to process a single PDF."""
    if config is None:
        config = ProcessingConfig()
    
    pipeline = PDFEnrichmentPipeline(config)
    return pipeline.process_pdf(pdf_path, output_path)

def batch_process_pdfs(input_dir: Path, 
                      output_dir: Path,
                      config: Optional[ProcessingConfig] = None) -> List[ProcessingStats]:
    """Process all PDFs in a directory."""
    if config is None:
        config = ProcessingConfig()
    
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PDFEnrichmentPipeline(config)
    
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return []
    
    logger.info(f"Processing {len(pdf_files)} PDFs from {input_dir}")
    
    results = []
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}_enriched.txt"
        try:
            _, stats = pipeline.process_pdf(pdf_path, output_path)
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue
    
    logger.success(f"Batch processing complete: {len(results)} files processed")
    return results

# ========================= CLI Interface =========================

def main():
    """CLI interface for PDF enrichment."""
    parser = argparse.ArgumentParser(description="Enrich financial PDFs with visual analysis")
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="Input PDF file or directory")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output file or directory")
    parser.add_argument("--gemini-key", type=str,
                       help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--llmwhisperer-key", type=str,
                       help="LLMWhisperer API key (or set LLMWHISPERER_API_KEY env var)")
    parser.add_argument("--mineru-path", type=Path,
                       help="Path to MinerU directory if not installed via pip")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                       help="Device for MinerU model")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                       help="DPI for PDF image extraction")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL,
                       help="Gemini model to use")
    
    args = parser.parse_args()
    
    # Create config
    config = ProcessingConfig(
        dpi=args.dpi,
        device=args.device,
        gemini_model=args.model,
        gemini_api_key=args.gemini_key,
        llmwhisperer_api_key=args.llmwhisperer_key,
        mineru_path=args.mineru_path
    )
    
    try:
        if args.input.is_file():
            # Single file processing
            output_path = args.output or args.input.with_suffix(".enriched.txt")
            enriched_text, stats = process_pdf(args.input, output_path, config)
            print(f"✅ Processed successfully: {stats}")
        
        elif args.input.is_dir():
            # Directory processing
            output_dir = args.output or args.input / "enriched"
            results = batch_process_pdfs(args.input, output_dir, config)
            total_tokens = sum(r.total_tokens for r in results)
            total_time = sum(r.processing_time for r in results)
            print(f"✅ Batch processing complete: {len(results)} files, {total_tokens} tokens, {total_time:.1f}s")
        
        else:
            print(f"❌ Input path not found: {args.input}")
            return 1
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())