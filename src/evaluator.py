from loguru import logger
from config_loader import *
from sklearn.metrics import classification_report

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
        y_pred = self.model_trainer.predict(X_test) # Assuming predict returns the predicted labels
        report = classification_report(y_test, y_pred, digits=5)
        logger.info(f"Classification report for {self.model_factory.model_name}:\n{report}") # Log the classification report
        
        return report