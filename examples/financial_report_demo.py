#!/usr/bin/env python3
"""
Financial Report Processing Demo
===============================

This script demonstrates how to use the DSPy optimization pipeline
on real financial documents, showing the complete workflow from
training to production inference.
"""

import json
import dspy
from typing import Dict, List, Any
from pathlib import Path

from src.config import configure_llm
from training.bootstrap_trainer import TripletExtractionOptimizer, OptimizedTripletExtractor


def load_financial_data() -> tuple:
    """Load financial text and existing extraction results."""
    
    # Example paths - adjust to your actual data location
    text_path = "data/financial_report.txt"
    extraction_path = "data/financial_extraction.json"
    
    try:
        with open(text_path, "r") as f:
            financial_text = f.read()
        
        with open(extraction_path, "r") as f:
            existing_extraction = json.load(f)
        
        return financial_text, existing_extraction
    except FileNotFoundError as e:
        print(f"⚠️  Sample data not found: {e}")
        return create_sample_data()


def create_sample_data() -> tuple:
    """Create sample financial data for demonstration."""
    
    sample_text = """
    Apple Inc. (AAPL) reported quarterly revenue of $89.5 billion for Q1 2024, 
    representing a 2% decline year-over-year. The company's iPhone revenue fell 
    to $69.7 billion, down 15% from the previous year due to challenging market 
    conditions in China. However, Services revenue grew 11.3% to $23.1 billion, 
    demonstrating the strength of Apple's ecosystem.
    
    The Federal Reserve's aggressive interest rate hikes have created headwinds 
    for technology stocks, with many investors rotating into defensive sectors. 
    Apple's forward P/E ratio of 24.8x suggests the market expects modest growth 
    despite near-term challenges.
    
    Looking ahead, management expressed cautious optimism about the upcoming 
    iPhone 15 launch and potential recovery in Chinese demand. The company 
    maintains a strong balance sheet with $166 billion in cash and equivalents.
    """
    
    sample_extraction = {
        "entities": [
            {
                "name": "Apple Inc.",
                "type": "Company",
                "brief": "Technology company",
                "description": "Consumer electronics and software company",
                "properties": {"ticker": "AAPL", "sector": "Technology"}
            },
            {
                "name": "Q1 2024 Revenue",
                "type": "Metric",
                "brief": "Quarterly revenue figure", 
                "description": "Total revenue for first quarter 2024",
                "properties": {"value": 89.5, "unit": "USD_bn", "period": "Q1_2024"}
            }
        ],
        "relationships": [
            {
                "source": "Apple Inc.",
                "target": "Q1 2024 Revenue",
                "rel_type": "REPORTS",
                "properties": {"confidence": 95},
                "description": "Company reported quarterly revenue"
            }
        ]
    }
    
    return sample_text, sample_extraction


def create_training_chunks(text: str, chunk_size: int = 1500) -> List[str]:
    """Split financial text into training chunks."""
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def create_training_examples(chunks: List[str], existing_extraction: Dict) -> List[Dict]:
    """Create training examples from chunks and existing extraction."""
    
    training_examples = []
    existing_entities = existing_extraction.get("entities", [])
    existing_relationships = existing_extraction.get("relationships", [])
    
    for i, chunk in enumerate(chunks):
        chunk_entities = []
        chunk_relationships = []
        
        # Find entities mentioned in chunk
        chunk_lower = chunk.lower()
        for entity in existing_entities:
            entity_name = entity.get("name", "").lower()
            if entity_name and entity_name in chunk_lower:
                chunk_entities.append(entity)
        
        # Find relationships for chunk entities
        chunk_entity_names = {e.get("name", "") for e in chunk_entities}
        for rel in existing_relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source in chunk_entity_names and target in chunk_entity_names:
                chunk_relationships.append(rel)
        
        if len(chunk_entities) >= 1:
            training_examples.append({
                "text": chunk,
                "entities": chunk_entities,
                "relationships": chunk_relationships,
                "chunk_id": i
            })
    
    return training_examples


def demonstrate_financial_processing():
    """Complete demonstration of financial document processing."""
    
    print("🏦 Financial Document Processing Demo")
    print("=" * 50)
    
    # Step 1: Load or create data
    print("📄 Loading financial data...")
    financial_text, existing_extraction = load_financial_data()
    
    print(f"✅ Document length: {len(financial_text):,} characters")
    print(f"✅ Existing entities: {len(existing_extraction.get('entities', []))}")
    print(f"✅ Existing relationships: {len(existing_extraction.get('relationships', []))}")
    
    # Step 2: Create training data
    print("\n🔪 Creating training chunks...")
    chunks = create_training_chunks(financial_text)
    training_examples = create_training_examples(chunks, existing_extraction)
    
    print(f"✅ Created {len(chunks)} text chunks")
    print(f"✅ Created {len(training_examples)} training examples")
    
    if training_examples:
        example = training_examples[0]
        print(f"\n📋 Sample training example:")
        print(f"   Text: {example['text'][:100]}...")
        print(f"   Entities: {len(example['entities'])}")
        print(f"   Relationships: {len(example['relationships'])}")
    
    # Step 3: Initialize and train
    print("\n🤖 Initializing optimizer...")
    optimizer = TripletExtractionOptimizer()
    
    # Convert to DSPy examples
    dspy_examples = []
    for example in training_examples:
        dspy_ex = dspy.Example(
            text=example["text"],
            domain_context="financial_research",
            expected_entities=example["entities"],
            expected_relationships=example["relationships"]
        ).with_inputs("text", "domain_context")
        dspy_examples.append(dspy_ex)
    
    # Step 4: Test base extraction
    print("\n🔍 Testing base extraction...")
    sample_text = training_examples[0]["text"] if training_examples else financial_text[:1000]
    base_result = optimizer.base_extractor(sample_text)
    
    print(f"Base extraction results:")
    print(f"   Entities: {len(base_result.get('entities', []))}")
    print(f"   Relationships: {len(base_result.get('relationships', []))}")
    
    # Step 5: Optimize (if we have training examples)
    if dspy_examples:
        print("\n🎯 Running Bootstrap optimization...")
        try:
            # Use smaller subset for demo
            train_subset = dspy_examples[:3] if len(dspy_examples) > 3 else dspy_examples
            optimizer.optimize_with_bootstrap(train_subset, max_bootstrapped_demos=2)
            
            # Test optimized extraction
            print("\n✨ Testing optimized extraction...")
            optimized_result = optimizer.extract_with_optimization(sample_text)
            
            print(f"Optimized extraction results:")
            print(f"   Entities: {len(optimized_result.get('entities', []))}")
            print(f"   Relationships: {len(optimized_result.get('relationships', []))}")
            
            # Save results
            output_file = "financial_demo_results.json"
            demo_results = {
                "base_extraction": base_result,
                "optimized_extraction": optimized_result,
                "metadata": {
                    "demo_type": "financial_report",
                    "training_examples": len(dspy_examples),
                    "optimization_method": "Bootstrap Few-Shot"
                }
            }
            
            with open(output_file, "w") as f:
                json.dump(demo_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Demo results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            print("📋 This is normal for demo data - in production, use more training examples")
    
    else:
        print("\n⚠️  No training examples created - using base extraction only")
    
    print("\n🎉 Financial processing demo completed!")
    
    return optimizer, base_result


def show_extraction_details(result: Dict[str, Any]):
    """Display detailed extraction results."""
    
    print("\n📊 Detailed Extraction Results:")
    print("=" * 40)
    
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    
    print(f"\n🏢 Entities ({len(entities)}):")
    for i, entity in enumerate(entities[:5], 1):  # Show first 5
        print(f"  {i}. {entity.get('name', 'Unknown')}")
        print(f"     Type: {entity.get('type', 'Unknown')}")
        print(f"     Description: {entity.get('description', 'N/A')[:60]}...")
        if entity.get('properties'):
            props = list(entity['properties'].items())[:2]  # First 2 properties
            print(f"     Properties: {dict(props)}")
        print()
    
    print(f"\n🔗 Relationships ({len(relationships)}):")
    for i, rel in enumerate(relationships[:5], 1):  # Show first 5
        print(f"  {i}. {rel.get('source', 'Unknown')} → {rel.get('target', 'Unknown')}")
        print(f"     Type: {rel.get('rel_type', 'Unknown')}")
        print(f"     Description: {rel.get('description', 'N/A')}")
        if rel.get('properties', {}).get('confidence'):
            print(f"     Confidence: {rel['properties']['confidence']}%")
        print()


if __name__ == "__main__":
    try:
        optimizer, results = demonstrate_financial_processing()
        
        if results:
            show_extraction_details(results)
            
        print("\n💡 Next Steps:")
        print("   1. Use real financial documents for better training")
        print("   2. Collect more training examples")
        print("   3. Experiment with different optimization methods")
        print("   4. Integrate with your document processing pipeline")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have set up your OpenAI API key and dependencies")