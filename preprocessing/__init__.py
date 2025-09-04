"""
PDF Preprocessing Module
========================

This module provides end-to-end PDF processing capabilities:
- High-quality text extraction using LLMWhisperer
- Figure/chart detection and cropping using MinerU
- Visual element analysis using Gemini Vision API
- Markdown enrichment with structured data from charts

The output enriched text can then be processed by the DSPy training
and inference pipelines for knowledge graph extraction.
"""

from .pdf_enricher import PDFEnrichmentPipeline, process_pdf

__all__ = [
    "PDFEnrichmentPipeline",
    "process_pdf",
]