import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/config.yaml"
PREPROCESSING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/preprocessing_config.yaml"
MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/model_config.yaml"

def load_config():
    """Load the main configuration file (config.yaml)."""
    # Makes sure that the file exists:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        # Makes sure that yaml file is in the correct format:
        if not isinstance(config, dict):
            raise ValueError("Configuration file must return a dictionary.")
        return config

def load_preprocessing_config():
    """Load the preprocessing configuration from config.yaml."""
    config = load_config()
    # Makes sure that the file exists:
    if not PREPROCESSING_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Preprocessing configuration file not found: {PREPROCESSING_CONFIG_PATH}")
    with open(PREPROCESSING_CONFIG_PATH, "r") as file:
        preprocessing_config = yaml.safe_load(file)
        # Makes sure that yaml file is in the correct format:
        if not isinstance(preprocessing_config, dict):
            raise ValueError("Preprocessing configuration must return a dictionary.")
        return preprocessing_config

def load_model_config():
    """Load parameters from the main configuration file (model_config.yaml)."""
    # Makes sure that the file exists:
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model configuration file not found: {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "r") as file:
        model_config = yaml.safe_load(file)
        # Makes sure that yaml file is in the correct format:
        if not isinstance(model_config, dict):
            raise ValueError("Model configuration must return a dictionary.")
        return model_config