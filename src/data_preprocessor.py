"""
Module for preprocessing data before model training.
"""
from pathlib import Path
import os
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from category_encoders import OneHotEncoder, TargetEncoder
from config_loader import load_config, load_preprocessing_config

class DataPreprocessor:
    """Class to preprocess data for model training."""
    
    def __init__(self, problem_type):
        """
        Initialize the DataCleaner.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.encoders = {}
        self.scalers = {}
        self.problem_type = problem_type
    
    
    def preprocess(self, df):
        """
        Preprocess the data for model training.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Preprocessed dataframe
        """
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Drop unnecessary columns
        df_processed = self._drop_columns(df_processed)
        
        # Encode categorical variables
        df_processed = self._encode_categorical_columns(df_processed)
        
        # Scale numerical variables
        df_processed = self._scale_numerical_columns(df_processed)
        
        # Save processed data
        dir = os.path.join(Path(__file__).resolve().parents[1], self.config['paths']['processed_data']+'processed'+self.problem_type+'_data.csv')
        df_processed.to_csv(dir, index=False)
        # Log progress
        logger.info(f"Data for {self.problem_type} problem saved to {dir}")

        
        return df_processed

    
    def _encode_categorical_columns(self, df):
        """Encode categorical variables."""
        if self.problem_type == 'regression':
            categorical_cols_not2encode = []
        elif self.problem_type == 'classification':
            categorical_cols_not2encode = self.config['data']['target_cat']
        categorical_cols = self.preprocessing_config['columns']['categorical']
        
        categorical_cols_2encode = [col for col in categorical_cols if col in df.columns and col not in categorical_cols_not2encode]

        if not categorical_cols_2encode:
            return df
            
        encoding_method = self.preprocessing_config['preprocessing']['categorical_encoding']['method']
        handle_unknown = self.preprocessing_config['preprocessing']['categorical_encoding']['handle_unknown']
        
        if encoding_method == 'one-hot':
            encoder = OneHotEncoder(cols=categorical_cols_2encode, handle_unknown=handle_unknown)
            df = encoder.fit_transform(df)
            self.encoders['categorical'] = encoder
            
        elif encoding_method == 'label':
            for col in categorical_cols_2encode:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
        elif encoding_method == 'target':
            # Will be implemented in the feature engineering step
            # as it requires the target variable
            pass
            
        elif encoding_method == 'frequency':
            for col in categorical_cols_2encode:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map)
                self.encoders[col] = freq_map
        
        return df
    
    def _scale_numerical_columns(self, df):
        """Scale numerical variables."""
        numerical_cols_2scale = self.preprocessing_config['columns']['numerical']
        numerical_cols_not2scale = self.config['data']['target_num']
        numerical_cols = [col for col in numerical_cols_2scale if col in df.columns and col not in numerical_cols_not2scale]
        
        if not numerical_cols:
            return df
            
        scaling_method = self.preprocessing_config['preprocessing']['numerical_scaling']['method']
        
        if scaling_method == 'none':
            return df
        
        # Create a copy of the numerical columns for scaling
        df_scaled = df.copy()
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
            
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        
        # Fit and transform the numerical columns
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers['numerical'] = scaler
        
        return df_scaled
    
    def _handle_outliers(self, df):
        """Handle outliers in numerical features based on the specified strategy."""
        if not self.preprocessing_config['cleaning']['handle_outliers']['enabled']:
            return df
        
        
        if self.preprocessing_config['cleaning']['handle_outliers']==None:
            return df
        
        method = self.preprocessing_config['cleaning']['handle_outliers']['method']
        iqr_multiplier = self.preprocessing_config['cleaning']['handle_outliers'].get('iqr_multiplier', 1.5)
        zscore_threshold = self.preprocessing_config['cleaning']['handle_outliers'].get('zscore_threshold', 3.0)
        outlier_columns = self.preprocessing_config['cleaning']['handle_outliers'].get('columns', [])
        
        for col in outlier_columns:
            if col in df.columns:
                if method == 'zscore':
                    from scipy.stats import zscore

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