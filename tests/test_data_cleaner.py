import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_loader import DataLoader
from data_cleaner import DataCleaner

def get_test_df():
    loader = DataLoader()
    loader.raw_data_path = "../data/bmarket.db"  # Change the raw data path as this script is not in the same directory as data_loader.py
    return loader.load_data()

def test_clean_data_real():
    df = get_test_df()
    cleaner = DataCleaner()
    cleaned = cleaner.clean_data(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty

def test_drop_columns():
    df = get_test_df()
    cleaner = DataCleaner()
    df = cleaner.drop_columns(df.copy())
    assert isinstance(df, pd.DataFrame)

def test_handle_unknown_values():
    df = get_test_df()
    cleaner = DataCleaner()
    df = cleaner.handle_unknown_values(df.copy())
    assert isinstance(df, pd.DataFrame)

def test_extract_age():
    df = get_test_df()
    cleaner = DataCleaner()
    df = cleaner.extract_age(df.copy())
    assert isinstance(df, pd.DataFrame)

def test_impute():
    # Create a dummy dataframe with 'Education Level' and NaN values, using realistic label types
    df = pd.DataFrame({
        'Education Level': ['illiterate', 'university degree', None, 'high school', None, 'illiterate'],
        'number': [1, 2, 3, None, 5, 6]
    })
    cleaner = DataCleaner()
    df = cleaner.impute(df.copy())
    
    assert isinstance(df, pd.DataFrame)
    # Check if there are any NaN values remaining in 'Education Level'
    assert not df['Education Level'].isnull().any()

def test_handle_negative_values():
    df = get_test_df()
    cleaner = DataCleaner()
    df = cleaner.handle_negative_values(df.copy())
    assert isinstance(df, pd.DataFrame)

def test_remove_columns():
    df = get_test_df()
    cleaner = DataCleaner()
    df = cleaner.remove_columns(df.copy())
    assert isinstance(df, pd.DataFrame)