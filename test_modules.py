#!/usr/bin/env python3
"""
Simple test script to verify all modules import correctly
"""

def test_imports():
    """Test that all modules can be imported without errors."""
    
    print("🧪 Testing module imports...")
    
    # Test core modules
    try:
        from src.signatures import TripletExtractionSignature
        print("✅ src.signatures")
    except Exception as e:
        print(f"❌ src.signatures: {e}")
    
    try:
        from src.config import get_openai_key  # Test non-DSPy function first
        print("✅ src.config")
    except Exception as e:
        print(f"❌ src.config: {e}")
        
    try:
        from src.metrics import triplet_metric
        print("✅ src.metrics")
    except Exception as e:
        print(f"❌ src.metrics: {e}")
    
    # Test that basic structure works
    try:
        from src.extractors import Entity, KnowledgeGraph
        entity = Entity(name="Test", type="Company", brief="Test entity", description="A test")
        print("✅ Pydantic models work")
    except Exception as e:
        print(f"❌ Pydantic models: {e}")
    
    print("\n🎯 Testing training module imports...")
    
    try:
        from training.bootstrap_trainer import create_sample_training_data
        print("✅ training.bootstrap_trainer")
    except Exception as e:
        print(f"❌ training.bootstrap_trainer: {e}")
    
    print("\n🚀 Testing inference module imports...")
    
    # These are CLI scripts, so just test they can be imported
    try:
        import inference.batch_processor
        print("✅ inference.batch_processor")
    except Exception as e:
        print(f"❌ inference.batch_processor: {e}")
        
    try:
        import inference.model_runner
        print("✅ inference.model_runner")
    except Exception as e:
        print(f"❌ inference.model_runner: {e}")
    
    print("\n📚 Testing examples...")
    
    try:
        from examples.financial_report_demo import create_sample_data
        print("✅ examples.financial_report_demo")
    except Exception as e:
        print(f"❌ examples.financial_report_demo: {e}")
    
    print("\n🎉 Module testing completed!")


def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    
    print("\n🔧 Testing basic functionality...")
    
    try:
        from src.extractors import Entity, Relationship, KnowledgeGraph
        
        # Test Pydantic models
        entity = Entity(
            name="Apple Inc.",
            type="Company", 
            brief="Technology company",
            description="Consumer electronics company",
            properties={"ticker": "AAPL"}
        )
        
        relationship = Relationship(
            source="Apple Inc.",
            target="Q4 Revenue", 
            type="REPORTS",
            description="Company reports quarterly revenue"
        )
        
        kg = KnowledgeGraph(
            entities=[entity],
            relationships=[relationship],
            scenarios=[]
        )
        
        print(f"✅ Created knowledge graph with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")


if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
    
    print("\n💡 Next steps:")
    print("   1. Install missing dependencies: pip install -r requirements.txt")
    print("   2. Set up OPENAI_API_KEY environment variable")
    print("   3. Run training pipelines with your data")
    print("   4. Use inference tools for production")