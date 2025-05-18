from loguru import logger
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from config_loader import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    """
    Trains and evaluates classification models.
    """
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.model = model_factory.model
        self.model_name = model_factory.model_name
        self.config = load_config()
        self.model_config = load_model_config()
        self.preprocessing_config = load_preprocessing_config()
        self.label_encoder = LabelEncoder() # Initialize label encoder since it is binary classification

    def train(self, X_train, y_train):
        if self.model is None:
            self.model = self.model_factory.build_model() # Build the model if not already done
        y_encoded = self.label_encoder.fit_transform(y_train) # Encode Subscription Status
        logger.info(f"Training {self.model_name} model")
        self.model.fit(X_train, y_encoded)
        return self

    def predict(self, X):
        """
        Predict labels for the given input features.
        Args:
            X: Input features
        Returns:
            Array of predicted labels (decoded if label encoder was used)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred_encoded = self.model.predict(X)
        # If label_encoder has classes_, decode, else return as is
        if hasattr(self.label_encoder, "classes_"):
            return self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred_encoded

    def cross_validate(self, X, y):
        cv_config = self.model_config['cross-validation']
        scores = cross_val_score(self.model_factory.build_model(), X, y, **cv_config)
        logger.info(f"Cross-validation scores for {self.model_factory.model_name}: {scores}")
        return scores
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        Tune hyperparameters based on model configuration.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            tuple: (best_params, best_score)
                - best_params: Dictionary of best found parameters
                - best_score: Best cross-validation score
        """
        model_name = self.model_factory.model_name
        model_config = self.model_config['models'][model_name]

        if not model_config.get('hyperparameter_tuning', {}).get('enabled', False):
            logger.info(f"Hyperparameter tuning is disabled for {model_name}")
            return None, None

        param_grid = model_config['hyperparameter_tuning']['param_grid']
        if not param_grid:
            raise ValueError("param_grid cannot be empty")

        if self.model is None:
            self.model = self.model_factory.build_model()

        search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            n_jobs=-1, 
            verbose=1
        )

        try:
            y_encoded = self.label_encoder.fit_transform(y_train)
            search.fit(X_train, y_encoded)
            logger.info(f"Best parameters found: {search.best_params_}")
            logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
            
            # Update the model with the best estimator
            self.model = search.best_estimator_
            
            return search.best_params_, search.best_score_
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
