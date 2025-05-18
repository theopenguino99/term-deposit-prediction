from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from loguru import logger
from config_loader import *

class ModelFactory():
    """
    A class that handles the creation, loading, and saving of various machine learning models
    for subscription status classification.
    Attributes:
        model_name (str): Name of the model to be created ('random_forest', 'xgboost', or 'MLP')
        model_params (dict): Parameters for the selected model
        model: The machine learning model instance
        config (dict): General configuration settings
        model_config (dict): Model-specific configuration settings
        preprocessing_config (dict): Preprocessing configuration settings
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_params = {}
        self.model = None
        self.config = load_config()
        self.model_config = load_model_config()
        self.preprocessing_config = load_preprocessing_config()
        

    def build_model(self):
        """
        Build the model based on the specified model name.
        Returns:
            model: The machine learning model instance
        """
        if self.model_name == "random_forest":
            self.model_params = self.model_config['models']['random_forest']['params']
            self.model = RandomForestClassifier(**self.model_params)
            logger.info(f"Building Random Forest model with params: {self.model_params}")

        
        elif self.model_name == "xgboost":
            self.model_params = self.model_config['models']['xgboost']['params']
            self.model = XGBClassifier(**self.model_params)
            logger.info(f"Building XGBoost model with params: {self.model_params}")
        
        elif self.model_name == "MLP":
            self.model_params = self.model_config['models']['MLP']['params']
            self.model = MLPClassifier(**self.model_params)
            logger.info(f"Building MLP model with params: {self.model_params}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        return self.model


    def save(self, filepath):
        """
        Save the model to a file.
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_params': self.model_params
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load the model from a file.
        Args:
            filepath (str): Path to load the model from
        Returns:
            self: The current instance of ModelFactory
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_params = model_data['model_params']
        logger.info(f"Model loaded from {filepath}")
        return self # In order to chain the load method with other methods
