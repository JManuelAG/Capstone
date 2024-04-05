import os
import pandas as pd
from sklearn.model_selection import train_test_split

class ImportData:
    def __init__(self):
        self.non_bot_folder = ["E13.csv", 'TFP.csv']
        self.bot_folder = ['FSF.csv', 'INT.csv', 'TWT.csv']


    def type_data(self, type_data_merged):
        # Check the type of data 
        if type_data_merged == 1:
            return "clean_merged.csv"
        else:
            return "clean_users.csv"

    def read_and_sample_data(self, base_path="../Data/cresci-2015.csv/", type_data_merged = 1, bot_ratio=[.2, .8], bot_fldr_ratio=[1, 1, 1]):
        bot_data_frames = pd.DataFrame()
        total_bot = 0

        # Check the type of data to use
        type_data = self.type_data(type_data_merged)

        # Read non-bot data
        for folder in self.non_bot_folder:
            # Read path
            file_path = os.path.join(base_path, folder, "clean", type_data)
            if os.path.exists(file_path):
                non_bot_data_frames = pd.read_csv(file_path)
                total_non_bot = len(non_bot_data_frames)
                
        # Read and sample bot data
        for idx, folder in enumerate(self.bot_folder):
            file_path = os.path.join(base_path, folder, "clean", type_data)
            if os.path.exists(file_path):
                if bot_fldr_ratio[idx] == 1:
                    df = pd.read_csv(file_path)
                    bot_data_frames = pd.concat([bot_data_frames,  df], ignore_index=True)
                total_bot = len(bot_data_frames)

        # Calculate available ratios
        total_samples = total_bot + total_non_bot
        required_bot_samples = total_samples * bot_ratio[1]
        required_non_bot_samples = total_samples * bot_ratio[0]

        # Determine how many rows to keep based on the desired ratio and availability
        if required_bot_samples > total_bot:  # If less bot data is available than desired
            # Adjust bot rows to match the actual bot ratio
            required_bot_samples = total_bot
            required_non_bot_samples = round((bot_ratio[0] * required_bot_samples) / bot_ratio[1])

        elif required_non_bot_samples > total_non_bot:  # If less bot data is available than desired
            # Adjust non-bot rows to match the actual bot ratio
            required_non_bot_samples = total_non_bot
            required_bot_samples = round((bot_ratio[1] * required_non_bot_samples) / bot_ratio[0])

        # Sample 
        non_bot_df = non_bot_data_frames.sample(n=required_non_bot_samples)
        bot_df = bot_data_frames.sample(n=required_bot_samples)

        # Merge the adjusted data
        final_df = pd.concat([non_bot_df, bot_df], ignore_index=True)
        return final_df
    
    def split_dataset(self, data, proportions = [.7,.15,.15], target='bot'):
        """
        Splits the dataset into training, testing, and validation sets based on the provided proportions.
        
        Parameters:
        - data: The DataFrame to split.
        - proportions: A list of three numbers representing the proportion of training, testing, and validation sets, respectively.
        - target: The name of the target feature.
        
        Returns:
        A dictionary with keys 'X_train', 'X_test', 'X_val', 'y_train', 'y_test', 'y_val' and their corresponding DataFrame or Series as values.
        """
        
        if sum(proportions) != 1:
            raise ValueError("Proportions must sum up to 1.")
        
        train_prop, test_prop, val_prop = proportions
        
        # Split data into training and temporary data (temp will be split into test and validation)
        train_data, temp_data, y_train, y_temp = train_test_split(
            data.drop(target, axis=1), data[target], test_size=(1-train_prop), stratify=data[target], random_state=42)
        
        # Calculate proportion of test set relative to temp data
        test_relative_prop = test_prop / (test_prop + val_prop)
        
        # Split temp data into testing and validation
        X_test, X_val, y_test, y_val = train_test_split(
            temp_data, y_temp, test_size=(1-test_relative_prop), stratify=y_temp, random_state=42)
        
        # Prepare the result dictionary
        result = {
            'X_train': train_data,
            'X_test': X_test,
            'X_val': X_val,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val
        }
        
        return result