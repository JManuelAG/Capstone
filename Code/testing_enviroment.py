import os
import pandas as pd
from import_data import ImportData
from evaluation import Evaluate
from feature_selection import FeatureSelection
from models_test import ModelTester

class TestEnvironment:
    def __init__(self, DATASET, BOT_FOLDERS, BOT_RATIO, MERGED_DATASET, TYPE_SELECTION, TRAIN_RATE, TEST_RATE, VAL_RATE, MODEL, FEATURES, MODEL_P, GRID_SEARCH):
        """
        Initialize the TestEnvironment object.
        
        Args:
        - DATASET (str): Dataset name.
        - BOT_FOLDERS (list): List of bot folders.
        - BOT_RATIO (list): List containing the bot ratio.
        - MERGED_DATASET (bool): Indicates whether the dataset is merged.
        - TYPE_SELECTION (str): Type of feature selection.
        - TRAIN_RATE (float): Proportion of training data.
        - TEST_RATE (float): Proportion of testing data.
        - VAL_RATE (float): Proportion of validation data.
        - MODEL (str): Name of the model to be tested.
        - FEATURES (int): Number of features to consider.
        - MODEL_P (dict): Dictionary containing model parameters.
        - GRID_SEARCH (bool): Indicates whether to perform grid search.
        """
        # ************************************ Input Validations ***************************************
        # Validating DATASET
        if DATASET not in ['cresci-2015', 'cresci-2017']:
            raise ValueError("Invalid DATASET. Choose either 'cresci-2015' or 'cresci-2017'.")

        # Validating BOT_FOLDERS
        if not all(isinstance(folder, int) for folder in BOT_FOLDERS):
            raise ValueError("BOT_FOLDERS should be a list of integers.")
        if len(BOT_FOLDERS) != 3:
            raise ValueError("BOT_FOLDERS should have exactly three elements.")
        if not all(folder in [0, 1] for folder in BOT_FOLDERS):
            raise ValueError("BOT_FOLDERS elements should be either 0 or 1.")

        # Validating BOT_RATIO
        if not isinstance(BOT_RATIO, list) or len(BOT_RATIO) != 2:
            raise ValueError("BOT_RATIO should be a list containing exactly two elements.")
        if not all(isinstance(ratio, float) for ratio in BOT_RATIO):
            raise ValueError("Elements of BOT_RATIO should be floats.")
        if not (0 <= BOT_RATIO[0] <= 1 and 0 <= BOT_RATIO[1] <= 1):
            raise ValueError("BOT_RATIO elements should be between 0 and 1.")
        if sum(BOT_RATIO) != 1:
            raise ValueError("BOT_RATIO elements should add up to 1.")

        # Validating MERGED_DATASET
        if not isinstance(MERGED_DATASET, bool):
            raise ValueError("MERGED_DATASET should be a boolean value.")

        # Validating TYPE_SELECTION
        if TYPE_SELECTION not in ['correlation', 'chi2', 'classifier']:
            raise ValueError("Invalid TYPE_SELECTION. Choose either 'correlation', 'chi2', or 'classifier'.")

        # Validating TRAIN_RATE, TEST_RATE, and VAL_RATE
        if not all(isinstance(rate, float) for rate in [TRAIN_RATE, TEST_RATE, VAL_RATE]):
            raise ValueError("TRAIN_RATE, TEST_RATE, and VAL_RATE should be floats.")
        if not (0 <= TRAIN_RATE <= 1 and 0 <= TEST_RATE <= 1 and 0 <= VAL_RATE <= 1):
            raise ValueError("TRAIN_RATE, TEST_RATE, and VAL_RATE should be between 0 and 1.")
        if TRAIN_RATE + TEST_RATE + VAL_RATE != 1:
            raise ValueError("TRAIN_RATE, TEST_RATE, and VAL_RATE should sum up to 1.")

        # Validating MODEL
        if MODEL != 'all' and MODEL not in ['decision_tree', 'knn', 'logistic_regression', 'svm']:
            raise ValueError("Invalid MODEL. Choose either 'all', 'decision_tree', 'knn', 'logistic_regression', or 'svm'.")

        # Validating FEATURES
        if FEATURES is not None and not isinstance(FEATURES, int):
            raise ValueError("FEATURES should be an integer or None.")
        if FEATURES not in [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            raise ValueError("Invalid number of FEATURES. Choose between 1 and 12 or None.")

        # Validating MODEL_P
        if not isinstance(MODEL_P, dict) and MODEL_P is not False:
            raise ValueError("MODEL_P should be a dictionary or False.")
        
        # ************************************ Finish Input Validations ***************************************
        self.DATASET = DATASET
        self.BOT_FOLDERS = BOT_FOLDERS
        self.BOT_RATIO = BOT_RATIO
        self.MERGED_DATASET = MERGED_DATASET
        self.TYPE_SELECTION = TYPE_SELECTION
        self.TRAIN_RATE = TRAIN_RATE
        self.TEST_RATE = TEST_RATE
        self.VAL_RATE = VAL_RATE
        self.MODEL = MODEL
        self.FEATURES = FEATURES
        self.MODEL_P = MODEL_P
        self.GRID_SEARCH = GRID_SEARCH

        # Initialize data and model tester
        self.importer = ImportData()
        self.data = self.importer.read_and_sample_data(dataset = DATASET, type_data_merged=self.MERGED_DATASET, bot_ratio=self.BOT_RATIO, bot_fldr_ratio=self.BOT_FOLDERS)
        self.selection = FeatureSelection(self.data)
        self.list_features = self.selection.select_features(type_selection=self.TYPE_SELECTION)
        self.splits = self.importer.split_dataset(data=self.data, proportions=[self.TRAIN_RATE, self.TEST_RATE, self.VAL_RATE])
        self.test_environment = ModelTester(self.splits, self.list_features)

    def save_results(self, model_parametres, test_metrics, val_metrics, MODEL):
        """
        Save the results to a CSV file.
        
        Args:
        - model_parametres (dict): Dictionary containing model parameters.
        - test_metrics (dict): Dictionary containing test metrics.
        - val_metrics (dict): Dictionary containing validation metrics.
        - MODEL (str): Name of the model.
        
        Returns:
        - df (DataFrame): DataFrame containing the saved results.
        """
        data = {
            "DATASET": self.DATASET,
            "BOT_FOLDERS": str(self.BOT_FOLDERS),
            "BOT_RATIO": str(self.BOT_RATIO),
            "MERGED_DATASET": self.MERGED_DATASET,
            "TYPE_SELECTION": self.TYPE_SELECTION,
            "TRAIN_RATE": self.TRAIN_RATE,
            "TEST_RATE": self.TEST_RATE,
            "VAL_RATE": self.VAL_RATE,
            "MODEL": MODEL,
            "FEATURES": self.FEATURES,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **model_parametres
        }

        # Create a DF that will store the current result and save at an Output path 
        df = pd.DataFrame([data])
        csv_file_name = f"{MODEL}_results.csv"
        file_path = f"../Outputs/{csv_file_name}"

        # Check that paths exists
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.drop_duplicates(keep='first', inplace=True)
            updated_df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

        return df

    def run_tests(self):
        """
        Run tests for the specified model.
        
        Returns:
        - results (dict): Dictionary containing the test results.
        """
        results = {}
        models = self.test_environment.models.keys() if self.MODEL == 'all' else [self.MODEL]

        # Test each of the desired models 
        for model in models:
            # Change parametres if specified
            if self.MODEL_P:
                self.test_environment.change_model_parameters(model_name=model, new_params=self.MODEL_P[model])

            # Optimize parametres if specified 
            if self.GRID_SEARCH:
                self.test_environment.grid_search(model_name=model, num_features=self.FEATURES)

            # Create a prediction
            model_parametres = self.test_environment.models[model].get_params()
            predictions = self.test_environment.predict_model(model_name=model, num_features=self.FEATURES)

            # Evaluate the Validation set
            val_evaluation = Evaluate(true_values=self.splits['y_val'], predicted_values=predictions['val_predictions'], predicted_probabilities=predictions['val_probabilities'])
            val_metrics = val_evaluation.get_all_metrics()

            # Evaluate the Test set
            test_evaluation = Evaluate(true_values=self.splits['y_test'], predicted_values=predictions['test_predictions'], predicted_probabilities=predictions['test_probabilities'])
            test_metrics = test_evaluation.get_all_metrics()

            # Save results on Excel 
            results[model] = self.save_results(model_parametres, test_metrics, val_metrics, MODEL=model)

        return results
