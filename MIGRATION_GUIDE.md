# Migration Guide: Old → New Structure

This document explains how to migrate from the old file structure to the new modular architecture.

## 🗂️ File Mapping

### Old Files → New Locations

| **Old File** | **New Location** | **Changes Made** |
|--------------|------------------|------------------|
| `DSPy_Triplet_Extraction_Optimization.py` | `training/bootstrap_trainer.py` | Modularized imports, updated structure |
| `mipro_pydantic_pipeline.py` | `training/mipro_trainer.py` | Simplified imports, kept core functionality |
| `mipro_optimization_pipeline.py` | `training/mipro_trainer_basic.py` | Basic MIPRO configuration |
| `mipro_llm_judge_pipeline.py` | `training/quality_evaluator.py` | LLM judge evaluation pipeline |
| `o3_triplet_extractor.py` | `inference/batch_processor.py` | Enhanced CLI interface |
| `run_saved_extractor.py` | `inference/model_runner.py` | Production inference tool |
| `nomura_dspy_example.py` | `examples/financial_report_demo.py` | Generalized example |
| `updated_triplet_prompt.md` | `prompts/triplet_extraction.md` | Moved to prompts folder |

### New Shared Modules

| **Module** | **Contains** | **Extracted From** |
|------------|-------------|-------------------|
| `src/config.py` | LLM configuration functions | All training files |
| `src/signatures.py` | DSPy signature definitions | All files with signatures |
| `src/extractors.py` | Core extractor classes | Multiple files |
| `src/metrics.py` | Evaluation metrics | Training files |

## 🔄 Import Changes

### Before (Old Structure)
```python
from DSPy_Triplet_Extraction_Optimization import TripletExtractionOptimizer
from mipro_pydantic_pipeline import configure_llm
import dspy
```

### After (New Structure)  
```python
from training.bootstrap_trainer import TripletExtractionOptimizer
from src.config import configure_llm
from src.extractors import TripletExtractor
import dspy
```

## 🚀 Usage Changes

### Training Commands

**Before:**
```bash
python DSPy_Triplet_Extraction_Optimization.py
python mipro_pydantic_pipeline.py
python mipro_llm_judge_pipeline.py
```

**After:**
```bash
python -m training.bootstrap_trainer
python -m training.mipro_trainer  
python -m training.quality_evaluator
```

### Inference Commands

**Before:**
```bash
python run_saved_extractor.py --program ./saved --input doc.txt
python o3_triplet_extractor.py
```

**After:**
```bash
python -m inference.model_runner --program ./saved --input doc.txt
python -m inference.batch_processor --input ./docs/ --output ./results/
```

## ✅ Migration Benefits

1. **🎯 Clear Organization**: Files grouped by purpose (training/inference/examples)
2. **🔧 Reduced Duplication**: Shared code extracted to `src/` modules  
3. **📚 Better Documentation**: Each module has clear responsibility
4. **🚀 Easier Maintenance**: Modular structure easier to update
5. **🐍 Python Standards**: Follows standard package structure
6. **⚡ Import Efficiency**: Shared modules loaded once

## 🛠️ Migration Steps

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Update Imports**: Use new module paths in your code
3. **Update Commands**: Use module syntax (`python -m module.name`)
4. **Test Functionality**: Run `python test_modules.py`
5. **Archive Old Files**: Old files moved to `archive_old_files/`

## 🚨 Breaking Changes

- **File Paths**: All files moved to new locations
- **Import Paths**: Module imports changed
- **CLI Commands**: Now use `-m` module syntax
- **Configuration**: Centralized in `src/config.py`

## 📞 Support

If you encounter issues during migration:
1. Check the new import paths in module `__init__.py` files
2. Ensure all dependencies are installed
3. Verify `OPENAI_API_KEY` is set
4. Run test script to verify imports work