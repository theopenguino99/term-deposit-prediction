import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

from models import ModelFactory
from trainer import ModelTrainer
from evaluator import ModelEvaluator

def get_dummy_data():
    X = np.random.rand(20, 4)
    y = np.random.choice(['class_a', 'class_b'], size=20)
    return X, y

def test_evaluator_summary():
    X, y = get_dummy_data()
    # Split into train/test
    X_train, X_test = X[:15], X[15:]
    y_train, y_test = y[:15], y[15:]

    # Build and train model
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    trainer.train(X_train, y_train)

    # Evaluate
    evaluator = ModelEvaluator(factory, trainer)
    accuracy, class_report, conf_matrix = evaluator.evaluate(X_test, y_test)

    # Print summary table of results
    print("\nEvaluation Summary Table:")
    print(f"{'accuracy':25}: {accuracy}")

    # Show classification report and confusion matrix
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    assert isinstance(accuracy, float)
    assert isinstance(class_report, str)
    assert hasattr(conf_matrix, "shape")