import os
from io import StringIO
import csv
import pandas as pd
from config_loader import *
from data_loader import DataLoader
from data_cleaner import DataCleaner
from sklearn.preprocessing import LabelBinarizer
from data_preprocessor import DataPreprocessor
from models import ModelFactory
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from loguru import logger


def main():
    """
    Main function to execute the data processing, model training, and evaluation pipeline.
    """
    # 1. Load data
    loader = DataLoader()
    df = loader.load_data()

    # 2. Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)

    # 3. Preprocess data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df_clean)

    # 4. Prepare features and labels
    X = df_processed.drop(columns=['Subscription Status'])
    y = df_processed['Subscription Status']

    # 5. Split data into train/test
    test_size = loader.config["data"]["test_size"]
    random_state = loader.config["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=random_state, stratify=y)

    # 6. Define models to train
    model_names = ["random_forest", "xgboost", "MLP"]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Prepare to plot all ROC curves on one figure
    plt.figure()
    colors = ['darkorange', 'green', 'blue']
    roc_curves = []

    for idx, model_name in enumerate(model_names):
        logger.info(f"Training and evaluating model: {model_name}")
        # 7. Build and train model
        factory = ModelFactory(model_name=model_name)
        trainer = ModelTrainer(factory)
        trainer.train(X_train, y_train)

        # 8. Evaluate model
        evaluator = ModelEvaluator(factory, trainer)
        report = evaluator.evaluate(X_test, y_test)

        # 9. Save classification report as CSV for each model
        report_lines = report.strip().split('\n')
        report_data = []
        for line in report_lines:
            if line.strip() and not line.startswith(('precision', 'recall', 'f1-score', 'support', 'accuracy')):
                row = [item for item in line.split() if item]
                if len(row) == 5:
                    report_data.append(row)
        if report_data:
            report_df = pd.DataFrame(report_data, columns=["class", "precision", "recall", "f1-score", "support"])
            report_path = os.path.join(data_dir, f"{model_name}_classification_report.csv")
            report_df.to_csv(report_path, index=False)
            logger.info(f"Saved classification report for {model_name} to {report_path}")

        # 10. Save the trained model using ModelFactory's save method to the models directory
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_save_path = os.path.join(models_dir, f"{model_name}_model.joblib")
        factory.save(model_save_path)
        logger.info(f"Saved trained model for {model_name} to {model_save_path}")

        # 11. Get ROC Curve data for this model
        if hasattr(trainer.model, "predict_proba"):
            y_score = trainer.model.predict_proba(X_test)
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            else:
                y_score = y_score[:, 1]
        else:
            y_score = trainer.model.decision_function(X_test)

        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        if y_test_bin.shape[1] == 1:
            y_test_bin = y_test_bin.ravel()

        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr, roc_auc, model_name, colors[idx % len(colors)]))

    # Plot all ROC curves on the same figure
    for fpr, tpr, roc_auc, model_name, color in roc_curves:
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - All Models')
    plt.legend(loc="lower right")
    roc_path = os.path.join(os.path.dirname(__file__), "../roc_curve_all_models.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Saved combined ROC curve for all models to {roc_path}")

if __name__ == "__main__":
    main()