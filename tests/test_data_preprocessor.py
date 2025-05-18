import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor

import pytest

@pytest.fixture
def real_df():
    loader = DataLoader()
    loader.raw_data_path = "../data/bmarket.db"  # Change the raw data path as this script is not in the same directory as data_loader.py
    df = loader.load_data()
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_data(df)
    return cleaned_df

def test_preprocess_real_data(real_df):
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess(real_df)
    assert isinstance(processed, pd.DataFrame)
    assert not processed.empty
    try:
        assert not processed.isna().any().any(), "NaN values found in processed dataset"
        
    except AssertionError as e:
        print("\nDataFrame Info:")
        print(processed.info())
        print("\nUnique values in 'Education Level':")
        print(processed['Education Level'].unique())
        raise e

def test_handle_outliers(real_df):
    preprocessor = DataPreprocessor()
    df_out = preprocessor.handle_outliers(real_df.copy())
    assert isinstance(df_out, pd.DataFrame)

def test_scale_numerical_columns(real_df):
    preprocessor = DataPreprocessor()
    df_scaled = preprocessor.scale_numerical_columns(real_df.copy())
    assert isinstance(df_scaled, pd.DataFrame)

def test_encode(real_df):
    preprocessor = DataPreprocessor()
    df_encoded = preprocessor.encode(real_df.copy())
    assert isinstance(df_encoded, pd.DataFrame)

