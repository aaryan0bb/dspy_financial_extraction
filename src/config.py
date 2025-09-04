#!/usr/bin/env python3
"""
Configuration and LLM Setup for Financial Extraction Pipeline
=============================================================

Centralized configuration management for API keys, model selection,
and DSPy LLM configuration across all pipeline components.
"""

import os
import logging
import dspy

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 20000

def get_openai_key():
    """Get OpenAI API key from environment or return empty string."""
    return os.getenv("OPENAI_API_KEY", "")

def configure_llm(model: str = DEFAULT_MODEL, 
                  temperature: float = DEFAULT_TEMPERATURE,
                  max_tokens: int = DEFAULT_MAX_TOKENS,
                  api_key: str = None):
    """Configure DSPy to use OpenAI with specified parameters."""
    
    # Set API key
    openai_key = api_key or get_openai_key()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Configure DSPy LLM
    lm = dspy.LM(f'openai/{model}', temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    
    logger.info(f"✅ DSPy configured with {model} (temp={temperature}, max_tokens={max_tokens})")
    return lm

def configure_openai_direct(model: str = DEFAULT_MODEL, api_key: str = None):
    """Configure OpenAI client directly for non-DSPy usage."""
    
    openai_key = api_key or get_openai_key()
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    
    os.environ["OPENAI_API_KEY"] = openai_key
    
    logger.info(f"✅ OpenAI configured with {model}")
    return openai_key