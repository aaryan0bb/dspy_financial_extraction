"""
Production Inference Tools
==========================

This package contains production-ready tools for running trained models:
- Batch processing of text documents
- Loading and running saved DSPy programs
- Production inference pipelines
"""

# Note: These modules are designed to be run as scripts
# They can be imported but are primarily CLI tools

__all__ = [
    "batch_processor",
    "model_runner",
]