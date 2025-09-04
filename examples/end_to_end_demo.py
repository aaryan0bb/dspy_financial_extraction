#!/usr/bin/env python3
"""
End-to-End PDF Processing and Knowledge Extraction Demo
=======================================================

This example demonstrates the complete pipeline:
1. PDF → Enriched Text (using preprocessing module)
2. Enriched Text → Knowledge Graphs (using DSPy training/inference)

Shows how to process financial PDFs from start to finish.
"""

import os
from pathlib import Path
from typing import Optional

from preprocessing.pdf_enricher import PDFEnrichmentPipeline, ProcessingConfig
from training.mipro_trainer import extract_knowledge_graph
from src.config import configure_llm


def demo_end_to_end_processing(pdf_path: Path, 
                              output_dir: Path,
                              gemini_key: Optional[str] = None,
                              llmwhisperer_key: Optional[str] = None,
                              openai_key: Optional[str] = None):
    """Complete end-to-end processing demo."""
    
    print("🚀 End-to-End Financial Document Processing Demo")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: PDF Enrichment
    print(f"\n📄 Step 1: Processing PDF with visual analysis...")
    print(f"   Input PDF: {pdf_path}")
    
    # Configure preprocessing
    config = ProcessingConfig(
        gemini_api_key=gemini_key,
        llmwhisperer_api_key=llmwhisperer_key,
        dpi=300,
        device="cpu"
    )
    
    # Process PDF
    pipeline = PDFEnrichmentPipeline(config)
    enriched_text_path = output_dir / f"{pdf_path.stem}_enriched.txt"
    
    try:
        enriched_text, stats = pipeline.process_pdf(pdf_path, enriched_text_path)
        
        print(f"   ✅ PDF processing complete!")
        print(f"   📊 Statistics:")
        print(f"      - Figures processed: {stats.figures_processed}")
        print(f"      - API tokens used: {stats.total_tokens:,}")
        print(f"      - Processing time: {stats.processing_time:.1f}s")
        print(f"      - Enriched text saved: {enriched_text_path}")
        
    except Exception as e:
        print(f"   ❌ PDF processing failed: {e}")
        print("   💡 Make sure you have set GEMINI_API_KEY and LLMWHISPERER_API_KEY")
        return
    
    # Step 2: Knowledge Extraction
    print(f"\n🧠 Step 2: Extracting knowledge graphs from enriched text...")
    
    try:
        # Configure DSPy
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        configure_llm()
        
        # Extract knowledge from a sample of the enriched text
        sample_text = enriched_text[:2000]  # Use first 2000 characters for demo
        knowledge_graph = extract_knowledge_graph(sample_text)
        
        print(f"   ✅ Knowledge extraction complete!")
        print(f"   📊 Extracted:")
        print(f"      - Entities: {len(knowledge_graph.entities)}")
        print(f"      - Relationships: {len(knowledge_graph.relationships)}")
        print(f"      - Scenarios: {len(knowledge_graph.scenarios)}")
        
        # Save results
        results_path = output_dir / f"{pdf_path.stem}_knowledge_graph.json"
        with open(results_path, 'w') as f:
            import json
            f.write(knowledge_graph.model_dump_json(indent=2))
        
        print(f"      - Results saved: {results_path}")
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        if knowledge_graph.entities:
            entity = knowledge_graph.entities[0]
            print(f"   🏢 Sample Entity: {entity.name}")
            print(f"      Type: {entity.type}")
            print(f"      Description: {entity.description}")
        
        if knowledge_graph.relationships:
            rel = knowledge_graph.relationships[0]
            print(f"   🔗 Sample Relationship: {rel.source} → {rel.target}")
            print(f"      Type: {rel.type}")
            print(f"      Description: {rel.description}")
        
    except Exception as e:
        print(f"   ❌ Knowledge extraction failed: {e}")
        print("   💡 Make sure you have set OPENAI_API_KEY")
        return
    
    # Step 3: Summary
    print(f"\n🎉 End-to-End Processing Complete!")
    print(f"   📈 Pipeline Performance:")
    print(f"      - PDF → Enriched Text: {stats.processing_time:.1f}s")
    print(f"      - Visual elements processed: {stats.figures_processed}")
    print(f"      - Knowledge entities extracted: {len(knowledge_graph.entities)}")
    print(f"      - All outputs saved to: {output_dir}")
    
    return {
        "enriched_text": enriched_text,
        "knowledge_graph": knowledge_graph,
        "processing_stats": stats
    }


def demo_with_sample_data():
    """Demo using sample/placeholder data if no real PDF available."""
    
    print("🧪 Demo with Sample Data (No Real PDF Processing)")
    print("=" * 50)
    
    # Sample enriched text that would come from PDF processing
    sample_enriched_text = """
    Apple Inc. (AAPL) Financial Performance Q1 2024
    
    Apple Inc. reported quarterly revenue of $119.6 billion for Q1 2024, representing a 2% increase year-over-year. The company's iPhone segment generated $69.7 billion in revenue.
    
    [VISUAL ELEMENT ANALYSIS]
    
    | Quarter | Revenue (B$) | Growth (%) | iPhone Rev (B$) |
    |---------|-------------|------------|-----------------|  
    | Q1 2023 | 117.2       | +5.0%      | 65.8            |
    | Q2 2023 | 94.8        | -2.5%      | 51.3            |
    | Q3 2023 | 81.8        | +1.4%      | 39.7            |
    | Q4 2023 | 89.5        | -1.0%      | 43.8            |
    | Q1 2024 | 119.6       | +2.1%      | 69.7            |
    
    Summary: The chart displays Apple's quarterly revenue progression over five quarters, showing seasonal patterns with Q1 consistently being the strongest quarter due to holiday sales. iPhone revenue follows similar patterns, representing approximately 58-60% of total revenue across quarters.
    
    Explanation: The data reveals Apple's continued dependence on iPhone sales for revenue generation, with Q1 2024 showing recovery in growth momentum after several quarters of decline or minimal growth.
    """
    
    # Extract knowledge from sample text
    try:
        configure_llm()
        knowledge_graph = extract_knowledge_graph(sample_enriched_text)
        
        print(f"✅ Sample processing complete!")
        print(f"📊 Extracted from sample enriched text:")
        print(f"   - Entities: {len(knowledge_graph.entities)}")
        print(f"   - Relationships: {len(knowledge_graph.relationships)}")
        
        # Show results
        for i, entity in enumerate(knowledge_graph.entities[:3], 1):
            print(f"   {i}. Entity: {entity.name} ({entity.type})")
            print(f"      Description: {entity.description}")
        
        for i, rel in enumerate(knowledge_graph.relationships[:3], 1):
            print(f"   {i}. Relationship: {rel.source} → {rel.target}")
            print(f"      Type: {rel.type}")
        
        return knowledge_graph
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure OPENAI_API_KEY is set")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-end processing demo")
    parser.add_argument("--pdf", type=Path, help="PDF file to process")
    parser.add_argument("--output", type=Path, default="./demo_results", 
                       help="Output directory")
    parser.add_argument("--sample", action="store_true", 
                       help="Run with sample data (no PDF processing)")
    
    args = parser.parse_args()
    
    if args.sample:
        # Run sample demo
        result = demo_with_sample_data()
    
    elif args.pdf and args.pdf.exists():
        # Run full end-to-end demo
        result = demo_end_to_end_processing(
            pdf_path=args.pdf,
            output_dir=args.output,
            gemini_key=os.getenv("GEMINI_API_KEY"),
            llmwhisperer_key=os.getenv("LLMWHISPERER_API_KEY"),
            openai_key=os.getenv("OPENAI_API_KEY")
        )
    
    else:
        print("❌ Please provide a PDF file with --pdf or run with --sample")
        print("\nUsage examples:")
        print("  python -m examples.end_to_end_demo --sample")
        print("  python -m examples.end_to_end_demo --pdf document.pdf --output ./results")
        print("\nRequired environment variables:")
        print("  - OPENAI_API_KEY (for knowledge extraction)")
        print("  - GEMINI_API_KEY (for visual analysis)")  
        print("  - LLMWHISPERER_API_KEY (for text extraction)")
    
    if result:
        print(f"\n💡 Next steps:")
        print(f"   1. Use the enriched text for training: python -m training.mipro_trainer")
        print(f"   2. Run batch processing: python -m preprocessing.pdf_enricher --input ./pdfs/")
        print(f"   3. Production inference: python -m inference.model_runner")