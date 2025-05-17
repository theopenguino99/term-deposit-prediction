from loguru import logger
from sklearn.model_selection import cross_val_score
from config_loader import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class ModelTrainer:
    """
    Trains and evaluates classification models.
    """
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.model = model_factory.model
        self.config = load_config()
        self.model_config = load_model_config()
        self.preprocessing_config = load_preprocessing_config()

    def train(self, X_train, y_train):
        self.model_factory.train(X_train, y_train)

    def train(self, X_train, y_train):
        if self.model is None:
            self.build_model()
        y_encoded = self.label_encoder.fit_transform(y_train)
        logger.info(f"Training {self.model_name} model")
        self.model.fit(X_train, y_encoded)
        return self

    def cross_validate(self, X, y, cv=5, scoring=None):
        if not scoring:
            scoring = self.model_factory.metric
        scores = cross_val_score(self.model_factory.build_model(), X, y, cv=cv, scoring=scoring)
        logger.info(f"Cross-validation scores for {self.model_factory.model_name}: {scores}")
        return scores

    def evaluate(self, X_test, y_test):
        results = self.model_factory.evaluate(X_test, y_test)
        logger.info(f"Evaluation results for {self.model_factory.model_name}: {results}")
        return results

    def save(self, filepath):
        self.model_factory.save(filepath)

    def load(self, filepath):
        self.model_factory.load(filepath)
    
    def tune_hyperparameters(self, X_train, y_train, param_grid, search_type="grid", cv=5, scoring=None, n_iter=10):
        """
        Tune hyperparameters using grid search or random search.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            search_type: "grid" for GridSearchCV, "random" for RandomizedSearchCV
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_iter: Number of iterations for RandomizedSearchCV

        Returns:
            best_params: Best found parameters
            best_score: Best cross-validation score
        """
        if self.model_factory.model is None:
            self.model_factory.build_model()
        model = self.model_factory.model

        if search_type == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif search_type == "random":
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=42)
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        y_encoded = self.model_factory.label_encoder.fit_transform(y_train)
        search.fit(X_train, y_encoded)

        logger.info(f"Best params: {search.best_params_}, Best score: {search.best_score_}")
        # Update the model in the factory with the best estimator
        self.model_factory.model = search.best_estimator_
        return search.best_params_, search.best_score_