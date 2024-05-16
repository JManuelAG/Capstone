import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class ModelTester:
    def __init__(self, data_dict, feature_importance):
        """
        Initialize the ModelTester object.
        
        Args:
        - data_dict (dict): Dictionary containing train, test, and validation datasets.
        - feature_importance (DataFrame): DataFrame containing feature importances.
        """
        self.X_train = data_dict['X_train']
        self.X_test = data_dict['X_test']
        self.X_val = data_dict['X_val']
        self.y_train = data_dict['y_train']
        self.y_test = data_dict['y_test']
        self.y_val = data_dict['y_val']
        self.feature_importance = feature_importance
        self.models = {}  # Dictionary to store models and their parameters
        
        # Initialize models with parameters from 'Parameters' folder
        self.load_models()

        # Createa a template parametre GRID for initial test of each model
        logistic_regression_params = {
            'penalty': ['l2', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [50, 100, 200, 500],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'class_weight': [None, 'balanced']
        }

        knn_params = {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 50],
            'p': [1, 2]
        }

        svm_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False],
            'probability': [True, False],
            'class_weight': [None, 'balanced']
        }

        decision_tree_params = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'class_weight': [None, 'balanced']
        }

        # Combine all parameter grids into a dictionary
        self.parameters_dict = {
            'logistic_regression': logistic_regression_params,
            'knn': knn_params,
            'svm': svm_params,
            'decision_tree': decision_tree_params
        }


    def load_models(self):
        """
        Load models with parameters from 'Parameters' folder.
        """
        parameters_folder = '../Parameters'
        for filename in os.listdir(parameters_folder):
            if filename.endswith('.pkl'):
                model_name = filename.split('.')[0]
                model = joblib.load(os.path.join(parameters_folder, filename))
                self.models[model_name] = model

    def change_model_parameters(self, model_name, new_params):
        """
        Change parameters of the specified model.
        
        Args:
        - model_name: Name of the model to change parameters for.
        - new_params: Dictionary containing new parameter values.
        """
        model = self.models[model_name]
        model.set_params(**new_params)
        self.models[model_name] = model

    def save_current_parameters(self, model_name):
        """
        Save the current parameters of the specified model to 'Parameters' folder.
        
        Args:
        - model_name: Name of the model to save parameters for.
        """
        parameters_folder = '../Parameters'
        joblib.dump(self.models[model_name], os.path.join(parameters_folder, f"{model_name}.pkl"))

    def fit_all_models(self, num_features=None):
        """
        Fit all models with the training data.
        """
        if num_features is None:
            for model_name, model in self.models.items():
                model.fit(self.X_train, self.y_train)
                self.models[model_name] = model
        else:
            features = self.feature_importance[:num_features]
            for model_name, model in self.models.items():
                model.fit(self.X_train[features], self.y_train)
                self.models[model_name] = model

    def grid_search(self, model_name, param_grid=None, scoring='f1', num_features=None, save_feature=False):
        """
        Perform grid search to fine-tune parameters for the specified model.
        
        Args:
        - model_name: Name of the model to perform grid search on.
        - param_grid: Dictionary specifying the range of parameters for grid search (default is None, which uses self.parameters_dict).
        - scoring: Scoring metric for grid search optimization (default is 'f1').
        - num_features: Number of features to consider in the grid search (default is None, which uses all features).
        """
        if param_grid is None:
            param_grid = self.parameters_dict[model_name]
            
        model = self.models[model_name]
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5)

        # Search according to the specified number of features 
        if num_features is None:
            # fit the models with all features
            self.fit_all_models()
            grid_search.fit(self.X_train, self.y_train)
        else:
            # Check if the number of features is valid
            if num_features <= 0 or num_features > len(self.X_train.columns):
                raise ValueError("Invalid number of features provided.")
            
            # fit the models with specified number of features
            self.fit_all_models(num_features=num_features)

            # Get the features to train
            features = self.feature_importance[:num_features]
            
            # Fit the model using the specified number of features
            grid_search.fit(self.X_train[features], self.y_train)

        best_params = grid_search.best_params_
        self.change_model_parameters(model_name, best_params)

        # Save the result if true
        if save_feature:
            self.save_current_parameters(model_name)

        return best_params
    

    def predict_model(self, model_name, num_features=None):
        """
        Generate predictions (class labels and probabilities) for the chosen model using test and validation data.
        
        Args:
        - model_name: Name of the model for which predictions are generated.
        
        Returns:
        - predictions_dict: Dictionary containing predicted class labels and probabilities for test and validation data.
                            Keys: 'test_predictions', 'test_probabilities', 'val_predictions', 'val_probabilities'
        """
        if model_name not in self.models:
            raise ValueError("Model not found in the dictionary.")
        
        model = self.models[model_name]
        
        if num_features is None:
            # Fit the models
            self.fit_all_models()

            # Predictions and probabilities on test data
            test_predictions = model.predict(self.X_test)
            test_probabilities = model.predict_proba(self.X_test)[:, 1]  # Extract probabilities of positive class (class 1)
            
            # Predictions and probabilities on validation data
            val_predictions = model.predict(self.X_val)
            val_probabilities = model.predict_proba(self.X_val)[:, 1]
        else:
            # Check if the number of features is valid
            if num_features <= 0 or num_features > len(self.X_train.columns):
                raise ValueError("Invalid number of features provided.")
            
            # Fit the models with number of features
            self.fit_all_models(num_features=num_features)

            # Get the features to train
            features = self.feature_importance[:num_features]

            # Predictions and probabilities on test data
            test_predictions = model.predict(self.X_test[features])
            test_probabilities = model.predict_proba(self.X_test[features])[:, 1]  # Extract probabilities of positive class (class 1)
            
            # Predictions and probabilities on validation data
            val_predictions = model.predict(self.X_val[features])
            val_probabilities = model.predict_proba(self.X_val[features])[:, 1]
                    
        # Construct the predictions dictionary
        predictions_dict = {
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'val_predictions': val_predictions,
            'val_probabilities': val_probabilities
        }
        
        return predictions_dict
