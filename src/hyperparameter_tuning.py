import os
import pandas as pd
from config_loader import *
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor
from models import ModelFactory
from trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from loguru import logger

def main():
    # 1. Load and clean data
    loader = DataLoader()
    df = loader.load_data()
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df_clean)

    # 2. Prepare features and labels
    X = df_processed.drop(columns=['Subscription Status'])
    y = df_processed['Subscription Status']

    # 3. Split data
    test_size = loader.config["data"]["test_size"]
    random_state = loader.config["data"]["random_state"]
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 4. Define models to tune
    model_names = ["random_forest", "xgboost", "MLP"]
    results = []

    for model_name in model_names:
        logger.info(f"Tuning hyperparameters for {model_name}")
        factory = ModelFactory(model_name=model_name)
        trainer = ModelTrainer(factory)
        # try:
        best_params, best_score = trainer.tune_hyperparameters(X_train, y_train)
        logger.info(f"Best params for {model_name}: {best_params}")
        logger.info(f"Best CV score for {model_name}: {best_score}")
        results.append({
            "model": model_name,
            "best_params": best_params,
            "best_score": best_score
        })
        # except Exception as e:
        #     logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")
        #     results.append({
        #         "model": model_name,
        #         "best_params": None,
        #         "best_score": None,
        #         "error": str(e)
        #     })

    # 5. Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(os.path.dirname(__file__), "..", "data", "hyperparameter_tuning_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Hyperparameter tuning results saved to {results_path}")

if __name__ == "__main__":
    main()