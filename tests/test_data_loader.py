import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_loader import DataLoader

def test_dataloader_init():
    """Ensure DataLoader initializes properly."""
    loader = DataLoader()
    assert hasattr(loader, "config")
    assert hasattr(loader, "preprocessing_config")

def test_load_data_file_not_found():
    """Check that a missing file raises FileNotFoundError."""
    loader = DataLoader()
    loader.raw_data_path = "non_existent.db"
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_load_from_sqlite():
    """Verify data is correctly loaded from the existing database."""
    loader = DataLoader()
    loader.raw_data_path = "../data/bmarket.db"  # Change the raw data path as this script is not in the same directory as data_loader.py
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns)  # Ensure it has columns
    assert len(df) > 0  # Ensure there are rows