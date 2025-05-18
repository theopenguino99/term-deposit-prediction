import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

import pytest
from models import ModelFactory
from trainer import ModelTrainer

def get_dummy_data():
    X = np.random.rand(20, 4)
    y = np.random.choice(['something', 'another thing'], size=20) # Example binary classification
    return X, y

def test_train_and_predict():
    X, y = get_dummy_data()
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    trainer.train(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)

def test_tune_hyperparameters_disabled():
    X, y = get_dummy_data()
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    # Assume config disables hyperparameter tuning for this test
    trainer.model_config['models']['random_forest']['hyperparameter_tuning']['enabled'] = False
    best_params, best_score = trainer.tune_hyperparameters(X, y)
    assert best_params is None and best_score is None

def test_tune_hyperparameters_empty_grid():
    X, y = get_dummy_data()
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    trainer.model_config['models']['random_forest']['hyperparameter_tuning']['enabled'] = True
    trainer.model_config['models']['random_forest']['hyperparameter_tuning']['param_grid'] = {}
    with pytest.raises(ValueError):
        trainer.tune_hyperparameters(X, y)

def test_cross_validate():
    X, y = get_dummy_data()
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    trainer.model_config['cross-validation'] = {'cv': 3}
    scores = trainer.cross_validate(X, y)
    assert hasattr(scores, '__iter__')
    assert len(scores) == 3

def test_trainer_predict():
    X, y = get_dummy_data()
    factory = ModelFactory(model_name="random_forest")
    trainer = ModelTrainer(factory)
    trainer.train(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)
    # Check that predictions are from the set of labels in y
    assert set(preds).issubset(set(y))