"""
Module for preprocessing data before model training.
"""
from pathlib import Path
import os
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from category_encoders import OneHotEncoder
from config_loader import load_config, load_preprocessing_config
from scipy.stats import zscore

class DataPreprocessor:
    """Class to preprocess data for model training."""
    
    def __init__(self):
        """
        Initialize the DataCleaner.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.encoders = {}
        self.scalers = {}
    
    
    def preprocess(self, df):
        """
        Preprocess the data for model training in this order:
        1. Handle outliers
        2. Scale numerical variables
        3. Encode categorical variables
        4. Save the processed data as a CSV file
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Preprocessed dataframe
        """
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.scale_numerical_columns(df_processed)
        df_processed = self.encode(df_processed)
        
        
        # Save processed data
        dir = os.path.join(Path(__file__).resolve().parents[1], self.config['paths']['processed_data']+'cleaned_and_processed_data.csv')
        df_processed.to_csv(dir, index=False)
        # Log progress
        logger.info(f"Cleaned and preprocessed data saved to {dir}")

        
        return df_processed
    
    def encode(self, df):
        """Encode categorical variables based on their type (ordinal or nominal)."""
        if not self.preprocessing_config['columns']['categorical']:
            return df
            
        ordinal_cols = self.preprocessing_config['columns']['categorical'].get('Ordinal', [])
        onehot_cols = self.preprocessing_config['columns']['categorical'].get('One-hot', [])

        # Check which columns exist in the dataframe
        ordinal_cols = [col for col in ordinal_cols if col in df.columns]
        onehot_cols = [col for col in onehot_cols if col in df.columns]

        # Handle ordinal columns with label encoding
        if ordinal_cols:
            for col in ordinal_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded ordinal column: {col}")

        # Handle nominal columns with one-hot encoding
        if onehot_cols:
            encoder = OneHotEncoder(cols=onehot_cols)
            df = encoder.fit_transform(df)
            self.encoders['onehot'] = encoder
            logger.info(f"One-hot encoded columns: {onehot_cols}")

        return df
    
    def handle_outliers(self, df):
        """Handle outliers in numerical features based on the specified strategy."""
        method = self.preprocessing_config['preprocessing']['handle_outliers']['method']
        iqr_multiplier = self.preprocessing_config['preprocessing']['handle_outliers'].get('iqr_multiplier', 1.5)
        zscore_threshold = self.preprocessing_config['preprocessing']['handle_outliers'].get('zscore_threshold', 3.0)
        outlier_columns = self.preprocessing_config['preprocessing']['handle_outliers'].get('columns', [])
        if self.preprocessing_config['preprocessing']['handle_outliers']['enabled']:
            for col in outlier_columns:
                if col in df.columns:
                    if method == 'zscore':
                        

                        z_scores = zscore(df[col].dropna())
                        outliers = abs(z_scores) > zscore_threshold
                        df.loc[outliers, col] = df[col].median()
                    
                    elif method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - iqr_multiplier * IQR
                        upper_bound = Q3 + iqr_multiplier * IQR
                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                        df.loc[outliers, col] = df[col].median()
        
        return df
    
    def scale_numerical_columns(self, df):
        """Scale numerical variables."""
        numerical_cols_2scale = self.preprocessing_config['preprocessing']['numerical_scaling']['columns']
        scaling_method = self.preprocessing_config['preprocessing']['numerical_scaling']['method']
        
        if numerical_cols_2scale:
            if scaling_method:
                # Create a copy of the numerical columns for scaling
                df_scaled = df.copy()
                
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                    
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                    
                elif scaling_method == 'robust':
                    scaler = RobustScaler()
                
                # Fit and transform the numerical columns
                df_scaled[numerical_cols_2scale] = scaler.fit_transform(df[numerical_cols_2scale])
                self.scalers['numerical'] = scaler
        
        return df_scaled
