import os
from io import StringIO
import csv
import pandas as pd
from config_loader import *
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from models import ModelFactory
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split


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
    validation_size = loader.config["data"].get("validation_size", 0.0)
    random_state = loader.config["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=random_state, stratify=y)

    # 6. Define models to train
    model_names = ["random_forest", "xgboost", "MLP"]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    for model_name in model_names:
        print(f"\nTraining and evaluating model: {model_name}")
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
            print(f"Saved classification report for {model_name} to {report_path}")

        # 10. Save the trained model using ModelFactory's save method to the models directory
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_save_path = os.path.join(models_dir, f"{model_name}_model.joblib")
        factory.save(model_save_path)
        print(f"Saved trained model for {model_name} to {model_save_path}")

if __name__ == "__main__":
    main()