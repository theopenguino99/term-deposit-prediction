import sys
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))
from data_loader import DataLoader
from data_cleaner import DataCleaner
from data_preprocessor import DataPreprocessor

import pytest
from models import ModelFactory

def test_model_factory_initialization():
    # Test initialization with valid model name
    factory = ModelFactory(model_name="random_forest")
    assert factory.model_name == "random_forest"
    assert factory.model_params == {}
    assert factory.model is None
    
    # Test initialization with no model name
    factory = ModelFactory()
    assert factory.model_name is None


def test_build_model():
    # Test building random forest model
    rf_factory = ModelFactory(model_name="random_forest") 
    rf_model = rf_factory.build_model()
    assert isinstance(rf_model, RandomForestClassifier)

    # Test building xgboost model
    xgb_factory = ModelFactory(model_name="xgboost")
    xgb_model = xgb_factory.build_model()
    assert isinstance(xgb_model, XGBClassifier)

    # Test building MLP model
    mlp_factory = ModelFactory(model_name="MLP")
    mlp_model = mlp_factory.build_model()
    assert isinstance(mlp_model, MLPClassifier)

def test_invalid_model():
    # Test invalid model name raises error
    with pytest.raises(ValueError):
        factory = ModelFactory(model_name="invalid_model")
        factory.build_model()

def test_save_load_model(tmp_path):
    # Test saving and loading model
    factory = ModelFactory(model_name="random_forest")
    model = factory.build_model()
    
    save_path = os.path.join(tmp_path, "test_model.joblib")
    factory.save(save_path)
    assert os.path.exists(save_path)

    # Test loading saved model
    new_factory = ModelFactory()
    loaded_factory = new_factory.load(save_path)
    assert loaded_factory.model_name == "random_forest"
    assert isinstance(loaded_factory.model, RandomForestClassifier)

def test_save_without_model():
    # Test saving without building model first
    factory = ModelFactory()
    with pytest.raises(ValueError):
        factory.save("test_model.joblib")

def test_load_nonexistent_model():
    # Test loading non-existent model file
    factory = ModelFactory()
    with pytest.raises(ValueError):
        factory.load("nonexistent_model.joblib")
