"""
Module for loading data from various sources.
"""

import os
import pandas as pd
import sqlite3
from loguru import logger
from pathlib import Path
from config_loader import *


class DataLoader:
    """Data loader class to load data from various sources."""
    
    def __init__(self):
        """
        Initialize the DataLoader.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.raw_data_path = load_raw_data()
        
    def load_data(self):
        """
        Load the data from the specified source.
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(self.raw_data_path):
            logger.error(f"Data file not found: {self.raw_data_path}")
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        # Determine file type and load accordingly
        file_extension = Path(self.raw_data_path).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(self.raw_data_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(self.raw_data_path)
        elif file_extension == '.json':
            df = pd.read_json(self.raw_data_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(self.raw_data_path)
        elif file_extension == '.db':
            df = self._load_from_sqlite(self.raw_data_path)
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
    
    def _load_from_sqlite(self, db_path):
        """
        Load data from an SQLite database.
        
        Args:
            db_path (str): Path to the SQLite database file
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        print(db_path)
        conn = sqlite3.connect(db_path)
        # Get the first table name from the SQLite database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_name = cursor.fetchone()[0]
        cursor.close()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def save_data(self, df, filepath):
        """
        Save data to the specified path.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            filepath (str): Path to save the data
        """
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        file_extension = Path(filepath).suffix.lower()
        
        if file_extension == '.csv':
            df.to_csv(filepath, index=False)
        elif file_extension in ['.xls', '.xlsx']:
            df.to_excel(filepath, index=False)
        elif file_extension == '.json':
            df.to_json(filepath, orient='records')
        elif file_extension == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            logger.error(f"Unsupported file format for saving: {file_extension}")
            raise ValueError(f"Unsupported file format for saving: {file_extension}")
        
        logger.info(f"Data saved successfully to {filepath}")