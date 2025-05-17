"""
Module for cleaning and preprocessing data.
"""
from config_loader import *
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.impute import KNNImputer


class DataCleaner:
    """Class to clean and preprocess data."""
    
    def __init__(self):
        """
        Initialize the DataCleaner.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.raw_data_path = self.config['paths']['raw_data']

        
    def clean_data(self, df):
        """
        Clean the data in this order:
        1a. Drop unnecessary columns
        1b. Handle unknown values
        2. Extract age from the 'Age' column
        3. Impute missing values
        4. Handle negative values in the 'Campaign Calls' column
        5. Remove columns that are not defined in the config file
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        df_cleaned = self.drop_columns(df_cleaned)
        df_cleaned = self.handle_unknown_values(df_cleaned)
        df_cleaned = self.extract_age(df_cleaned)
        df_cleaned = self.impute(df_cleaned)
        df_cleaned = self.handle_negative_values(df_cleaned)
        df_cleaned = self.remove_columns(df_cleaned)
        
        
        return df_cleaned
    
    def drop_columns(self, df):
        """
        Drop specified columns from the dataframe.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with dropped columns
        """
        if (self.preprocessing_config['cleaning']['drop_columns']['enabled'] and 
            self.preprocessing_config['cleaning']['drop_columns']['columns'] != None):
            columns_to_drop = self.preprocessing_config['cleaning']['drop_columns']['columns']
        
            # Drop specified columns
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)
                else:
                    logger.error(f"Column '{col}' specified in config file not found in DataFrame")
                    raise ValueError(f"Column '{col}' specified in config file not found in DataFrame")
                
        return df
    
    def handle_unknown_values(self, df):
        """
        Handle unknown values in the dataframe.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with handled unknown values
        """
        if not self.preprocessing_config['cleaning']['handle_unknown_values']['enabled']:
            return df
        
        columns_to_handle = self.preprocessing_config['cleaning']['handle_unknown_values']['columns']
        
        # Replace 'unknown' with NaN
        for col in columns_to_handle:
            if col not in df.columns:
                logger.error(f"Column '{col}' specified in config file not found in DataFrame")
                raise ValueError(f"Column '{col}' specified in config file not found in DataFrame")
            else:
                df[col] = df[col].replace('unknown', np.nan)
        return df
    
    def extract_age(self, df):
        """
        Extract age from the 'Age' column.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with extracted age
        """
        if 'Age' in df.columns:
            df['Age'] = df['Age'].str.replace('years', '').str.strip()
            df['Age'] = pd.to_numeric(df['Age'])
        
        return df
    

    def impute(self, df):
        """
        Impute missing values in the dataframe based on column type (we only have NaN values in the categorical data to impute).
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with imputed values
        """
        if not self.preprocessing_config['cleaning']['impute']['enabled']:
            return df

        # Get columns to impute and validate against handle_unknown_values columns
        columns_to_impute = self.preprocessing_config['cleaning']['impute']['columns']
        handle_unknown_cols = self.preprocessing_config['cleaning']['handle_unknown_values']['columns']
        
        if not set(columns_to_impute).issubset(set(handle_unknown_cols)):
            error_msg = "Columns to impute must be a subset of handle_unknown_values columns in config"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get imputation configurations
        cat_strategy = self.preprocessing_config['cleaning']['impute']['strategy']
        
        # Get column types
        cols = self.preprocessing_config['columns']

        # Only impute columns that actually have NaN values
        columns_with_nan = df.columns[df.isna().any()].tolist()
        columns_to_process = list(set(columns_to_impute) & set(columns_with_nan))

        # Impute categorical columns
        for col in set(columns_to_process) & set(cols):
            if cat_strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif cat_strategy == 'constant':
                const_val = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['constant_value']
                df[col] = df[col].fillna(const_val)
            elif cat_strategy == 'knn':
                # KNN imputation
                neighbors = self.preprocessing_config['cleaning']['impute']['neighbors']
                imputer = KNNImputer(n_neighbors=neighbors)
                df[col] = imputer.fit_transform(df[[col]])
            elif cat_strategy == 'random':
                # Random value imputation from column values
                random_value = df[col].dropna().sample(n=1).iloc[0]
                df[col] = df[col].fillna(random_value)

        return df
    
    def handle_negative_values(self, df):
        """
        Handle negative values in the 'Campaign Calls' column.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with removed or absolute of negative values
        """
        if self.preprocessing_config['cleaning']['handle_negative_values']['enabled']:
            if 'Campaign Calls' in df.columns:
                strategy = self.preprocessing_config['cleaning']['handle_negative_values']['strategy']
                if strategy == 'remove':
                    df = df[df['Campaign Calls'] >= 0]
                elif strategy == 'absolute':
                    df['Campaign Calls'] = df['Campaign Calls'].abs()
            else:
                logger.error("Column 'Campaign Calls' not found in DataFrame")
                raise ValueError("Column 'Campaign Calls' not found in DataFrame")
        
        return df
    
    def remove_columns(self, df):
        """
        Remove specified columns from the dataframe according to columns that are not defined in the config file.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with removed columns
        """
        columns_to_keep = (
            self.preprocessing_config['columns']['categorical']['Ordinal'] +
            self.preprocessing_config['columns']['categorical']['One-hot'] +
            self.preprocessing_config['columns']['numerical'] +
            self.preprocessing_config['columns']['others']
        )
        # Keep only columns defined in the config file
        columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
        
        if columns_to_remove:
            logger.info(f"Removing columns not defined in config: {columns_to_remove}")
            df.drop(columns=columns_to_remove, inplace=True)
        else:
            logger.info("No columns to remove, all columns are defined in the config!")
            
        return df
