import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/config.yaml"
PREPROCESSING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/preprocessing_config.yaml"
MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/model_config.yaml"

def load_config():
    """Load the main configuration file (config.yaml)."""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("Configuration file must return a dictionary.")
        return config

def load_preprocessing_config():
    """Load the preprocessing configuration from config.yaml."""
    config = load_config()
    preprocessing_path = CONFIG_PATH.parent.parent / config['pipeline']['preprocessing_config']
    with open(preprocessing_path, "r") as file:
        preprocessing_config = yaml.safe_load(file)
        if not isinstance(preprocessing_config, dict):
            raise ValueError("Preprocessing configuration must return a dictionary.")
        return preprocessing_config

def load_model_config():
    """Load parameters from the main configuration file (model_config.yaml)."""
    with open(MODEL_CONFIG_PATH, "r") as file:
        model_config = yaml.safe_load(file)
        if not isinstance(model_config, dict):
            raise ValueError("Model configuration must return a dictionary.")
        return model_config

def load_raw_data():
    """Load the raw_data directory (a .db SQL data base)."""
    config = load_config()
    preprocessing_path = CONFIG_PATH.parent.parent / config['paths']['raw_data']
    return preprocessing_path