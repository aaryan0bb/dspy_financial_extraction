#!/usr/bin/env python3
"""
Nomura Document Processing with DSPy Optimization
=================================================

This script demonstrates how to use your existing Nomura data and manual 
extraction results to train and optimize a DSPy-based triplet extractor.
"""

import json
import dspy
from typing import Dict, List, Any
from DSPy_Triplet_Extraction_Optimization import (
    TripletExtractionOptimizer, 
    OptimizedTripletExtractor
)


def load_nomura_data() -> tuple:
    """Load the Nomura text and existing extraction results."""
    
    # Load the source text
    with open("graph_structures/chunking_strategies/inputs/Nomura_Nomura Quant Insights_20250707_enriched.txt", "r") as f:
        nomura_text = f.read()
    
    # Load existing extraction results as gold standard
    with open("graph_structures/extracted-entities-1-v6/nomura_cross_asset_entities_relationships.json", "r") as f:
        existing_extraction = json.load(f)
    
    return nomura_text, existing_extraction


def create_training_chunks(text: str, chunk_size: int = 1500) -> List[str]:
    """Split the Nomura text into training chunks."""
    
    # Split by paragraphs first, then combine into chunks
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


def create_gold_standard_examples(chunks: List[str], existing_extraction: Dict) -> List[Dict]:
    """Create gold standard examples by mapping entities/relationships to text chunks."""
    
    gold_examples = []
    existing_entities = existing_extraction.get("entities", [])
    existing_relationships = existing_extraction.get("relationships", [])
    
    # For each chunk, find relevant entities and relationships
    for i, chunk in enumerate(chunks):
        chunk_entities = []
        chunk_relationships = []
        
        # Find entities mentioned in this chunk
        chunk_lower = chunk.lower()
        for entity in existing_entities:
            entity_name = entity.get("name", "").lower()
            if entity_name and entity_name in chunk_lower:
                chunk_entities.append(entity)
        
        # Find relationships where both source and target are in chunk entities
        chunk_entity_names = {e.get("name", "") for e in chunk_entities}
        for rel in existing_relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source in chunk_entity_names and target in chunk_entity_names:
                chunk_relationships.append(rel)
        
        # Only create examples for chunks with meaningful content
        if len(chunk_entities) >= 2:
            gold_examples.append({
                "text": chunk,
                "entities": chunk_entities,
                "relationships": chunk_relationships,
                "chunk_id": i
            })
    
    return gold_examples


def demonstrate_optimization_with_nomura():
    """Complete demonstration using Nomura data."""
    
    print("🏦 Nomura Document DSPy Optimization Demo")
    print("=" * 50)
    
    # Step 1: Load data
    print("📄 Loading Nomura data...")
    nomura_text, existing_extraction = load_nomura_data()
    
    print(f"✅ Loaded document ({len(nomura_text):,} characters)")
    print(f"✅ Existing extraction: {len(existing_extraction.get('entities', []))} entities")
    
    # Step 2: Create training chunks
    print("\n🔪 Creating training chunks...")
    chunks = create_training_chunks(nomura_text)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Step 3: Create gold standard examples
    print("\n🏆 Creating gold standard examples...")
    gold_examples = create_gold_standard_examples(chunks, existing_extraction)
    print(f"✅ Created {len(gold_examples)} training examples")
    
    # Show example
    if gold_examples:
        example = gold_examples[0]
        print(f"\n📋 Sample training example:")
        print(f"   Text length: {len(example['text'])} chars")
        print(f"   Entities: {len(example['entities'])}")
        print(f"   Relationships: {len(example['relationships'])}")
        print(f"   Sample entities: {[e.get('name', '') for e in example['entities'][:3]]}")
    
    # Step 4: Initialize DSPy optimizer
    print("\n🤖 Initializing DSPy optimizer...")
    optimizer = TripletExtractionOptimizer()
    
    # Convert to DSPy examples
    training_examples = []
    for gold in gold_examples:
        example = dspy.Example(
            text=gold["text"],
            domain_context="financial_research",
            expected_entities=gold["entities"],
            expected_relationships=gold["relationships"]
        ).with_inputs("text", "domain_context")
        training_examples.append(example)
    
    # Step 5: Test base extraction
    print("\n🔍 Testing base extraction on sample...")
    sample_text = gold_examples[0]["text"][:1000] + "..."
    base_result = optimizer.base_extractor(sample_text)
    
    print(f"Base extraction results:")
    print(f"   Entities found: {len(base_result.get('entities', []))}")
    print(f"   Relationships found: {len(base_result.get('relationships', []))}")
    print(f"   Quality score: {base_result.get('quality_assessment', {}).get('quality_score', 'N/A')}")
    
    # Step 6: Optimize with Bootstrap Few-Shot
    print("\n🎯 Optimizing with Bootstrap Few-Shot...")
    print("   This may take several minutes...")
    
    try:
        # Use subset for faster demo
        train_subset = training_examples[:5] if len(training_examples) > 5 else training_examples
        optimized_extractor = optimizer.optimize_with_bootstrap(train_subset, max_bootstrapped_demos=4)
        
        print("✅ Optimization completed!")
        
        # Step 7: Test optimized extraction
        print("\n✨ Testing optimized extraction...")
        optimized_result = optimizer.extract_with_optimization(sample_text)
        
        print(f"Optimized extraction results:")
        print(f"   Entities found: {len(optimized_result.get('entities', []))}")
        print(f"   Relationships found: {len(optimized_result.get('relationships', []))}")
        print(f"   Quality score: {optimized_result.get('quality_assessment', {}).get('quality_score', 'N/A')}")
        
        # Step 8: Compare results
        print("\n📊 Comparison Summary:")
        base_entities = len(base_result.get('entities', []))
        opt_entities = len(optimized_result.get('entities', []))
        base_rels = len(base_result.get('relationships', []))
        opt_rels = len(optimized_result.get('relationships', []))
        
        print(f"   Entities: {base_entities} → {opt_entities} ({opt_entities - base_entities:+d})")
        print(f"   Relationships: {base_rels} → {opt_rels} ({opt_rels - base_rels:+d})")
        
        # Step 9: Process full document
        print("\n📄 Processing full Nomura document...")
        full_result = process_full_document(nomura_text, optimizer)
        
        # Save results
        output_file = "nomura_dspy_optimized_extraction.json"
        with open(output_file, "w") as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Full extraction saved to: {output_file}")
        print(f"📈 Final results:")
        print(f"   Total entities: {len(full_result.get('entities', []))}")
        print(f"   Total relationships: {len(full_result.get('relationships', []))}")
        print(f"   Processing time: {full_result.get('metadata', {}).get('processing_time', 'N/A')}")
        
        return optimizer, full_result
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None, None


def process_full_document(text: str, optimizer: TripletExtractionOptimizer) -> Dict[str, Any]:
    """Process the complete Nomura document with the optimized extractor."""
    
    import time
    start_time = time.time()
    
    # Split into manageable chunks
    chunks = create_training_chunks(text, chunk_size=2000)
    
    all_entities = {}  # Use dict to deduplicate by name
    all_relationships = []
    all_scenarios = []
    
    print(f"   Processing {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 200:  # Skip very short chunks
            continue
        
        try:
            result = optimizer.extract_with_optimization(chunk)
            
            if "error" not in result:
                # Collect entities (deduplicate by name)
                for entity in result.get("entities", []):
                    name = entity.get("name", "")
                    if name:
                        all_entities[name] = entity
                
                # Collect relationships and scenarios
                all_relationships.extend(result.get("relationships", []))
                all_scenarios.extend(result.get("scenarios", []))
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            print(f"   Error in chunk {i + 1}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    return {
        "entities": list(all_entities.values()),
        "relationships": all_relationships,
        "scenarios": all_scenarios,
        "metadata": {
            "extraction_method": "DSPy_Optimized",
            "source_document": "Nomura_Nomura Quant Insights_20250707_enriched.txt",
            "processing_time": f"{processing_time:.2f} seconds",
            "chunks_processed": len(chunks),
            "unique_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "total_scenarios": len(all_scenarios),
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }


def compare_with_manual_extraction():
    """Compare DSPy results with manual extraction."""
    
    # Load both extractions
    with open("graph_structures/extracted-entities-1-v6/nomura_cross_asset_entities_relationships.json", "r") as f:
        manual_extraction = json.load(f)
    
    try:
        with open("nomura_dspy_optimized_extraction.json", "r") as f:
            dspy_extraction = json.load(f)
    except FileNotFoundError:
        print("❌ DSPy extraction file not found. Run the optimization first.")
        return
    
    print("\n🔍 Comparison: Manual vs DSPy Extraction")
    print("=" * 45)
    
    manual_entities = manual_extraction.get("entities", [])
    dspy_entities = dspy_extraction.get("entities", [])
    manual_rels = manual_extraction.get("relationships", [])
    dspy_rels = dspy_extraction.get("relationships", [])
    
    print(f"Entities:")
    print(f"   Manual: {len(manual_entities)}")
    print(f"   DSPy:   {len(dspy_entities)}")
    print(f"   Difference: {len(dspy_entities) - len(manual_entities):+d}")
    
    print(f"\nRelationships:")
    print(f"   Manual: {len(manual_rels)}")
    print(f"   DSPy:   {len(dspy_rels)}")
    print(f"   Difference: {len(dspy_rels) - len(manual_rels):+d}")
    
    # Entity overlap analysis
    manual_entity_names = {e.get("name", "") for e in manual_entities}
    dspy_entity_names = {e.get("name", "") for e in dspy_entities}
    
    overlap = manual_entity_names & dspy_entity_names
    manual_only = manual_entity_names - dspy_entity_names
    dspy_only = dspy_entity_names - manual_entity_names
    
    print(f"\nEntity Overlap Analysis:")
    print(f"   Common entities: {len(overlap)}")
    print(f"   Manual only: {len(manual_only)}")
    print(f"   DSPy only: {len(dspy_only)}")
    
    if manual_only:
        print(f"   Sample manual-only: {list(manual_only)[:3]}")
    if dspy_only:
        print(f"   Sample DSPy-only: {list(dspy_only)[:3]}")
    
    # Quality comparison
    dspy_metadata = dspy_extraction.get("metadata", {})
    print(f"\nProcessing Efficiency:")
    print(f"   Processing time: {dspy_metadata.get('processing_time', 'N/A')}")
    print(f"   Chunks processed: {dspy_metadata.get('chunks_processed', 'N/A')}")


if __name__ == "__main__":
    # Run the complete demonstration
    optimizer, results = demonstrate_optimization_with_nomura()
    
    if results:
        # Compare with manual extraction
        compare_with_manual_extraction()
        
        print("\n🎉 Demonstration completed!")
        print("\n💡 Key Takeaways:")
        print("   1. DSPy automatically optimizes prompts based on your data")
        print("   2. Quality improves with more training examples")
        print("   3. The system learns domain-specific patterns")
        print("   4. Processing becomes more consistent and scalable")
        print("\n🚀 Next steps:")
        print("   1. Collect more training examples for better optimization")
        print("   2. Experiment with different optimization strategies")
        print("   3. Integrate into your GraphRAG pipeline")
        print("   4. Set up continuous learning from feedback") 