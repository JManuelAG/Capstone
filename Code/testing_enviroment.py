import os
import pandas as pd
from import_data import ImportData
from evaluation import Evaluate
from feature_selection import FeatureSelection
from models_test import ModelTester

class TestEnvironment:
    def __init__(self, DATASET, BOT_FOLDERS, BOT_RATIO, MERGED_DATASET, TYPE_SELECTION, TRAIN_RATE, TEST_RATE, VAL_RATE, MODEL, FEATURES, MODEL_P, GRID_SEARCH):
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

        df = pd.DataFrame([data])
        csv_file_name = f"{MODEL}_results.csv"
        file_path = f"../Outputs/{csv_file_name}"

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.drop_duplicates(keep='first', inplace=True)
            updated_df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

        return df

    def run_tests(self):
        results = {}
        models = self.test_environment.models.keys() if self.MODEL == 'all' else [self.MODEL]

        for model in models:
            if self.MODEL_P:
                self.test_environment.change_model_parameters(model_name=model, new_params=self.MODEL_P[model])

            if self.GRID_SEARCH:
                self.test_environment.grid_search(model_name=model, num_features=self.FEATURES)

            model_parametres = self.test_environment.models[model].get_params()
            predictions = self.test_environment.predict_model(model_name=model, num_features=self.FEATURES)

            val_evaluation = Evaluate(true_values=self.splits['y_val'], predicted_values=predictions['val_predictions'], predicted_probabilities=predictions['val_probabilities'])
            val_metrics = val_evaluation.get_all_metrics()

            test_evaluation = Evaluate(true_values=self.splits['y_test'], predicted_values=predictions['test_predictions'], predicted_probabilities=predictions['test_probabilities'])
            test_metrics = test_evaluation.get_all_metrics()

            results[model] = self.save_results(model_parametres, test_metrics, val_metrics, MODEL=model)

        return results
