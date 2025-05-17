import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from config_loader import load_config, load_preprocessing_config, load_model_config

def test_load_config():
    """Ensure config.yaml loads correctly as a dictionary."""
    config = load_config()
    assert isinstance(config, dict)
    assert "pipeline" in config  

def test_load_preprocessing_config():
    """Ensure preprocessing config loads correctly."""
    preprocessing_config = load_preprocessing_config()
    assert isinstance(preprocessing_config, dict)

def test_load_model_config():
    """Ensure model config loads correctly."""
    model_config = load_model_config()
    assert isinstance(model_config, dict)
