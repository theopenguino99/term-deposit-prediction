from loguru import logger
from sklearn.model_selection import cross_val_score
from config_loader import *

class ModelEvaluator:
    """
    Handles cross-validation and evaluation for classification models.
    """
    def __init__(self, model_factory):
        self.config = load_config()
        self.model_config = load_model_config()
        self.model_factory = model_factory
        self.metric = self.model_config['evaluation_classification']['primary_metric']

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
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def evaluate(self, X_test, y_test):
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
        results['classification_report'] = classification_report(y_test, y_pred)
        results['confusion_matrix'] = confusion_matrix(y_encoded, y_pred_encoded)
        return results