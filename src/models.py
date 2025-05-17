from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib
import os
from loguru import logger
from config_loader import *

class ModelFactory():
    """Factory for creating Subscription Status classification models."""
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model_params = {}
        self.model = None
        self.label_encoder = LabelEncoder()
        self.config = load_config()
        self.model_config = load_model_config()
        self.preprocessing_config = load_preprocessing_config()
        

    def build_model(self):
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
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'model_params': self.model_params
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.model_params = model_data['model_params']
        logger.info(f"Model loaded from {filepath}")
        return self # In order to chain the load method with other methods
