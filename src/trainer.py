import os
import joblib
from loguru import logger
from config_loader import load_config, load_preprocessing_config, load_model_config
from sklearn.model_selection import train_test_split
from plant_type_stage_classification_models import (
    PlantTypeStageClassifier, EnsemblePlantClassifier, #AdaptivePlantClassifier
)
from temperature_regression_models import (
    TemperatureRegressionModel, #AdaptiveTemperatureRegressor , 
    DeepTemperatureRegressor
)

class ModelTrainer:
    """
    Trains machine learning models for both temperature prediction and plant type-stage classification.
    """
    
    def __init__(self):
        """
        Initialize the model trainer with configuration.
        """
        self.config = load_config()
        self.preprocessing_config = load_preprocessing_config()
        self.model_config = load_model_config()
        self.pipeline_config = self.config['pipeline']
        self.data_config = self.config['data']
        self.models_dir = self.config['paths']['models_dir']
        self.use_cross_validation = self.pipeline_config['use_cross_validation']
        self.n_folds = self.pipeline_config['n_folds']
        logger.info("ModelTrainer initialized with configuration")

    def prepare_data(self, features_data, target_column):
        """
        Prepare the data for training and testing.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the target column
            
        Returns:
            Tuple containing training and testing data
        """
        # Extract features and target
        X = features_data.drop(columns=[target_column])
        y = features_data[target_column]
        
        # Split data into training and testing sets
        test_size = self.data_config['test_size']
        random_state = self.data_config['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # If validation set is required
        validation_size = self.data_config['validation_size']
        if validation_size > 0:
            train_size = 1 - validation_size
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=train_size, random_state=random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_temperature_models(self, features_data):
        """
        Train regression models to predict temperature.
        
        Args:
            features_data (DataFrame): DataFrame with features
            target_column (str): Name of the temperature column
            
        Returns:
            Dictionary containing trained models and their performance metrics
        """
        logger.info("Starting temperature prediction model training")
        target_column = self.data_config['target_num']

        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from model_config
        enabled_models = [
            model_name for model_name, model_config in self.model_config['models'].items()
            if model_config['enabled'] == True
        ]
        
        results = {}
        primary_metric = self.model_config['evaluation_regression']['primary_metric']
        lower_is_better = self.model_config['evaluation_regression']['lower_is_better']
        best_metric = float('inf') if lower_is_better else float('-inf')
        best_model = None
        
        # Train standard models
        logger.opt(colors=True).info("<red>Training standard temperature prediction models</red>")
        for model_name in enabled_models:
            
            logger.info(f"Training {model_name} for temperature prediction")
            
            # Create model
            model_params = self.model_config['models'][model_name].get('params', {})
            model = TemperatureRegressionModel(model_name=model_name, model_params=model_params)
    
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams_config = self.model_config['models'][model_name].get('hyperparameter_tuning', {})
            if hyperparams_config.get('enabled', False):
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                hyperparams = hyperparams_config.get('param_grid', {})
                if hyperparams:
                    # Perform tuning
                    model, best_params = model.hyperparameter_tuning(
                        X_train, y_train, param_grid=hyperparams, cv=self.n_folds
                    )
            else:
                logger.info(f"Skipping hyperparameter tuning for {model_name}")

            # Save model
            model_path = os.path.join(self.models_dir, f"REGRESSION_{model_name}.pkl")
            joblib.dump(model,model_path)
            
            # Record results
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'model_path': model_path
            }
            
            # Update best model if needed
            if (lower_is_better and metrics[primary_metric] < best_metric) or \
               (not lower_is_better and metrics[primary_metric] > best_metric):
                best_metric = metrics[primary_metric]
                best_model = model_name
        
        # Train NN model if enabled
        if self.model_config['advanced_models']['deep_temperature']['enabled']:
            logger.opt(colors=True).info("<red>Training Deep Temperature Regressor</red>")
            deep_model = DeepTemperatureRegressor()
            deep_model.fit(X_train, y_train)
            deep_metrics = deep_model.evaluate(X_test, y_test)
            
            # Save deep model
            deep_path = os.path.join(self.models_dir, f"REGRESSION_deep_.pkl")
            joblib.dump(deep_model, deep_path)
            
            results['deep_temp'] = {
                'model': deep_model,
                'metrics': deep_metrics,
                'model_path': deep_path
            }
                
            if (lower_is_better and deep_metrics[primary_metric] < best_metric) or \
            (not lower_is_better and deep_metrics[primary_metric] > best_metric):
                best_metric = deep_metrics[primary_metric]
                best_model = 'deep_temp'
        
        # Log best model
        logger.info(f"Best temperature prediction model: {best_model} with RMSE: {best_metric}")
        
        return results, best_model
    
    def train_plant_type_stage_models(self, features_data):
        
        logger.info("Starting plant type-stage classification model training")
        target_column = self.data_config['target_cat']
        
        # Prepare data
        data_splits = self.prepare_data(features_data, target_column)
        if len(data_splits) == 6:
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        else:
            X_train, X_test, y_train, y_test = data_splits
            X_val, y_val = None, None
        
        # Get enabled models from model_config
        enabled_models = [
            model_name for model_name, model_config in self.model_config['models'].items()
            if model_config['enabled'] == True
        ]
        
        results = {}
        primary_metric = self.model_config['evaluation_classification']['primary_metric']
        lower_is_better = self.model_config['evaluation_classification']['lower_is_better']
        best_metric = float('inf') if lower_is_better else float('-inf')
        best_model = None
        
        # Train standard models
        logger.opt(colors=True).info("<red>Training standard classification prediction models</red>")
        for model_name in enabled_models:

            logger.info(f"Training {model_name} for plant type-stage classification")
            
            # Create model
            model_params = self.model_config['models'][model_name].get('params', {})
            model = PlantTypeStageClassifier(model_name=model_name, model_params=model_params)
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Perform hyperparameter tuning if enabled
            hyperparams_config = self.model_config['models'][model_name].get('hyperparameter_tuning', {})
            if hyperparams_config.get('enabled', False):
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                hyperparams = hyperparams_config.get('param_grid', {})
                if hyperparams:
                    # Perform tuning
                    model, best_params = model.hyperparameter_tuning(
                        X_train, y_train, param_grid=hyperparams, cv=self.n_folds
                    )
            else:
                logger.info(f"Skipping hyperparameter tuning for {model_name}")

            # Save model
            model_path = os.path.join(self.models_dir, f"CLASSIFICATION_{model_name}.pkl")
            joblib.dump(model,model_path)
            
            # Record results
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'model_path': model_path
            }
            
            # Update best model if needed
            if (lower_is_better and metrics[primary_metric] < best_metric) or \
                (not lower_is_better and metrics[primary_metric] > best_metric):
                best_metric = metrics[primary_metric]
                best_model = model_name
        
        # Train Ensemble models if enabled
        if self.model_config['advanced_models']['ensemble_plant']['enabled']:
            logger.info("Training Ensemble Plant Classifier")
            ensemble_model = EnsemblePlantClassifier()
            ensemble_model.fit(X_train, y_train)
            ensemble_metrics = ensemble_model.evaluate(X_test, y_test)
            
            # Save ensemble model
            ensemble_path = os.path.join(self.models_dir, f"CLASSIFICATION_ensemble.pkl")
            joblib.dump({'model': ensemble_model, 'label_encoder': ensemble_model.label_encoder}, ensemble_path)
            
            results['ensemble_plant'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'model_path': ensemble_path
            }
            
            # Update best model if any advanced model is better
            if (lower_is_better and ensemble_metrics[primary_metric] < best_metric) or \
                (not lower_is_better and ensemble_metrics[primary_metric] > best_metric):
                best_metric = ensemble_metrics[primary_metric]
                best_model = 'ensemble_plant'
        
        # Log best model
        logger.info(f"Best plant type-stage model: {best_model} with F1: {best_metric}")
        
        return results, best_model