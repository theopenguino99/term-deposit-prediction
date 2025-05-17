import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib
import os
from loguru import logger
from collections import Counter
from config_loader import load_model_config

class BaseTypeStageClassifier:
    """Base class for shared functionality in plant type-stage classification models."""
    
    def __init__(self, model_name=None, model_params=None, scaler=None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.scaler = scaler
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_config = load_model_config()
        self.metric = load_model_config()['evaluation_classification']['primary_metric']
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        y_encoded = self.label_encoder.transform(y_test)
        y_pred = self.predict(X_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        results = {}
        if self.metric == 'accuracy':
            results['accuracy'] = accuracy_score(y_encoded, y_pred_encoded)
        if self.metric == 'f1':
            results['f1_macro'] = f1_score(y_encoded, y_pred_encoded, average='macro')
            results['f1_weighted'] = f1_score(y_encoded, y_pred_encoded, average='weighted')
        if self.metric == 'precision':
            results['precision_macro'] = precision_score(y_encoded, y_pred_encoded, average='macro')
        if self.metric == 'recall':
            results['recall_macro'] = recall_score(y_encoded, y_pred_encoded, average='macro')
        if self.metric is None:
            raise ValueError("Classification metric not specified in model_config.yaml file.")
        
        # Add detailed classification report
        results['classification_report'] = classification_report(y_test, y_pred)
        
        # Add confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_encoded, y_pred_encoded)
        
        return results
    def save(self, filepath):
        """Save the model and label encoder to disk."""
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
        """Load the model and label encoder from disk."""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.model_params = model_data['model_params']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    


class PlantTypeStageClassifier(BaseTypeStageClassifier):
    """Class for plant type-stage classification models in the agricultural pipeline."""
    
    def __init__(self, model_name, model_params=None, scaler=None):
        super().__init__(model_name, model_params, scaler)
    
    def build_model(self):
        """Build the model based on model_name."""
        if self.model_name == "random_forest":
            model = RandomForestClassifier(**self.model_params)
        elif self.model_name == "gradient_boosting":
            model = GradientBoostingClassifier(**self.model_params)
        elif self.model_name == "xgboost":
            model = XGBClassifier(**self.model_params)
        elif self.model_name == "ridge_regression":
            model = RidgeClassifier(**self.model_params)
        elif self.model_name == "linear_regression":
            model = LogisticRegression(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        self.model = model
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        if self.model is None:
            self.build_model()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        logger.info(f"Training {self.model_name} model for plant type-stage classification")
        logger.info(f"Class distribution: {Counter(y_train)}")
        self.model.fit(X_train, y_encoded)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Make probability predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        return self.model.predict_proba(X)
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, n_jobs=-1, method='grid'):
        """Perform hyperparameter tuning for the model."""
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        if param_grid is None:
            # Define default param grid based on model type
            if self.model_name in load_model_config['models']:
                param_grid = load_model_config['models'][self.model_name]['hyperparameter_tuning']['param_grid']
            else:
                logger.warning(f"No default param grid for {self.model_name}. Using empty grid.")
                param_grid = {}
        
        # Create a base model
        base_model = self.build_model()
        
        # Set up the search
        scoring = 'f1_macro' if self.metric == 'f1' else self.metric
        if method == 'grid':
            search = GridSearchCV(base_model, param_grid, cv=cv, n_jobs=n_jobs,
                                 scoring=scoring)
        else:  # random search
            search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_jobs=n_jobs,
                                       scoring=scoring, n_iter=20)
        
        logger.info(f"Performing {method} search for hyperparameter tuning")
        search.fit(X, y_encoded)
        
        # Update model with best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        
        # Update model parameters and rebuild model
        if hasattr(search, 'best_estimator_'):
            self.model = search.best_estimator_
        else:
            # Extract best parameters and update model_params
            if self.scaler is not None:
                # For pipeline, need to extract classifier params
                best_params = {k.replace('classifier__', ''): v 
                             for k, v in search.best_params_.items() 
                             if k.startswith('classifier__')}
            else:
                best_params = search.best_params_
                
            self.model_params.update(best_params)
            self.build_model()
            self.model.fit(X, y_encoded)
        
        return self.model, search.best_params_


class EnsemblePlantClassifier(BaseTypeStageClassifier, BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier that combines multiple models for plant type-stage classification
    using voting or stacking techniques.
    """
    
    def __init__(self):
        super().__init__()
        config = load_model_config()['advanced_models']['ensemble_plant']
        random_state = load_model_config()['common']['random_state']
        RFC_n_estimators = load_model_config()['models']['random_forest']['params']['n_estimators']
        GBC_n_estimators = load_model_config()['models']['gradient_boosting']['params']['n_estimators']
        XGB_n_estimators = load_model_config()['models']['xgboost']['params']['n_estimators']
        
        self.base_models = [
            RandomForestClassifier(n_estimators=RFC_n_estimators, random_state=random_state),
            GradientBoostingClassifier(n_estimators=GBC_n_estimators, random_state=random_state),
            XGBClassifier(n_estimators=XGB_n_estimators, random_state=random_state)
        ]
        self.voting = config['params']['voting']
        self.weights = config['params']['weights']
        self.trained_models = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Train all base models on the same data."""
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Train all base models
        self.trained_models = []
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            model.fit(X, y_encoded)
            self.trained_models.append(model)
        
        return self.model
    
    def predict(self, X):
        """Make predictions using voting from all base models."""
        if self.trained_models is None:
            raise ValueError("Models have not been trained yet.")
        
        if self.voting == 'hard':
            # Get predictions from each model
            predictions = np.array([model.predict(X) for model in self.trained_models])
            
            # Transpose to get predictions per sample
            predictions = predictions.T
            
            # Use majority voting
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x, weights=self.weights).argmax(), 
                axis=1, 
                arr=predictions
            )
        else:  # soft voting
            # Get probability predictions
            probas = self.predict_proba(X)
            final_pred = np.argmax(probas, axis=1)
        
        # Convert back to original labels
        return self.label_encoder.inverse_transform(final_pred)
    
    def predict_proba(self, X):
        """Make probability predictions using averaging from all base models."""
        if self.trained_models is None:
            raise ValueError("Models have not been trained yet.")
        
        # Get probability predictions from each model
        all_probas = []
        for model in self.trained_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)
        
        # Average probabilities (with weights if provided)
        if self.weights:
            weights = np.array(self.weights)
            weights = weights / weights.sum()  # Normalize weights
            final_probas = np.average(np.array(all_probas), axis=0, weights=weights)
        else:
            final_probas = np.mean(np.array(all_probas), axis=0)
        
        return final_probas