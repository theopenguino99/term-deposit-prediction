"""
Module for cleaning and preprocessing data.
"""
from config_loader import load_config, load_preprocessing_config


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
        
    def clean_data(self, df):
        """
        Clean and preprocess the data.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        # Apply each cleaning step
        df_cleaned = self.clean_nutrient_sensors(df_cleaned)
        df_cleaned = self._map_labels_to_lowercase(df_cleaned)
        df_cleaned = self._handle_negative_values(df_cleaned)
        df_cleaned = self._handle_duplicates(df_cleaned)
        df_cleaned = self._handle_missing_values(df_cleaned)
        df_cleaned = self._handle_outliers(df_cleaned)
        
        
        return df_cleaned

    def clean_nutrient_sensors(self, df):
        """
        Clean and extract numerical values from nutrient sensor columns.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with cleaned nutrient sensor columns
        """
        if not self.preprocessing_config['cleaning']['clean_Nutrient_Sensor']['enabled']:
            return df
        
        
        nutrient_columns = self.preprocessing_config['cleaning']['clean_Nutrient_Sensor']['columns']
        
        for col in nutrient_columns:
            if col in df.columns:
                df[col] = df[col].str.extract('(\d+)', expand=False).astype(float)
        
        return df
        
    def _map_labels_to_lowercase(self, df):
        """Map capitalized labels in specified columns to lowercase."""
        if not self.preprocessing_config['cleaning']['map_labels_to_lowercase']['enabled']:
            return df
        
        
        columns_to_map = self.preprocessing_config['cleaning']['map_labels_to_lowercase']['columns']
        
        for col in columns_to_map:
            if col in df.columns:
                df[col] = df[col].str.lower()
        
        return df
    
    def _handle_negative_values(self, df):
        """Handle negative values in specified columns."""
        if not self.preprocessing_config['cleaning']['handle_negative_values']['enabled']:
            return df
        
        
        columns_to_handle = self.preprocessing_config['cleaning']['handle_negative_values']['columns']
        strategy = self.preprocessing_config['cleaning']['handle_negative_values']['strategy']
        
        for col in columns_to_handle:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    if strategy == 'remove':
                        df = df[df[col] >= 0]
                    elif strategy == 'absolute':
                        df[col] = df[col].abs()
        
        return df
    
    def _handle_duplicates(self, df):
        """Handle duplicate rows in the dataframe."""
        if not self.preprocessing_config['cleaning']['handle_duplicates']['enabled']:
            return df
        
        
        # Identify columns to ignore when detecting duplicates
        ignore_cols = self.preprocessing_config['cleaning']['handle_duplicates']['ignore_columns']
        cols_to_check = [col for col in df.columns if col not in ignore_cols]
        
        # Count duplicates
        n_duplicates = df.duplicated(subset=cols_to_check).sum()
        
        if n_duplicates > 0:
            # Keep only the first occurrence of duplicates
            keep_option = self.preprocessing_config['cleaning']['handle_duplicates']['keep']
            df = df.drop_duplicates(subset=cols_to_check, keep=keep_option)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe."""
        if not self.preprocessing_config['cleaning']['handle_missing_values']['enabled']:
            return df
        
        
        # Get missing value configurations
        num_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['numerical']['strategy']
        cat_strategy = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['strategy']
        
        # Get column lists
        num_cols = self.preprocessing_config['columns']['numerical']
        cat_cols = self.preprocessing_config['columns']['categorical']
        dt_cols = self.preprocessing_config['columns']['datetime']
        
        # Report missing values
        missing_info = df.isnull().sum()
        
        # Handle missing numerical values
        for col in num_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if num_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif num_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif num_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif num_strategy == 'constant':
                    const_val = self.preprocessing_config['cleaning']['handle_missing_values']['numerical']['constant_value']
                    df[col] = df[col].fillna(const_val)
        
        # Handle missing categorical values
        for col in cat_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                if cat_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_strategy == 'constant':
                    const_val = self.preprocessing_config['cleaning']['handle_missing_values']['categorical']['constant_value']
                    df[col] = df[col].fillna(const_val)
        
        # Report missing values after imputation
        missing_info_after = df.isnull().sum()
        
        return df
    
    
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