from loguru import logger
from config_loader import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class ModelEvaluator:
    """
    Handles cross-validation and evaluation for classification models.
    Attributes:
        model_factory (ModelFactory): Instance of ModelFactory for model creation
        model_trainer (ModelTrainer): Instance of ModelTrainer for training the model
        label_encoder (LabelEncoder): Label encoder for encoding labels
        metric (str): Metric used for evaluation
    """
    def __init__(self, model_factory, model_trainer):
        self.config = load_config()
        self.model_config = load_model_config()
        self.model_factory = model_factory
        self.model_trainer = model_trainer
        self.label_encoder = self.model_trainer.label_encoder

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the specified metric.
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        Returns:
            dict: Evaluation results including accuracy, precision, recall, and F1 score
        """
        y_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model_trainer.predict(X_test) # Assuming predict returns the predicted labels
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        accuracy = round(accuracy_score(y_encoded, y_pred_encoded), 5)
        report = classification_report(y_test, y_pred, digits=5)
        matrix = confusion_matrix(y_test, y_pred)
        
        return accuracy, report, matrix