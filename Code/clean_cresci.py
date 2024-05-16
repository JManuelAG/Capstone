import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


class clean_cresci_2015:
    def clean_data(self, base_directory = "../Data/cresci-2015.csv/"):
        """
        Clean data for Cresci 2015 dataset.

        Args:
            base_directory (str): Base directory containing dataset folders.

        Returns:
            None
        """
        # List all folders in the base directory
        folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

        # Loop through each folder and process datasets
        for folder in folders:
            tweets_path = os.path.join(base_directory, folder, "tweets.csv")
            users_path = os.path.join(base_directory, folder, "users.csv")

            # Now you can process these datasets as needed
            print(f"Processing datasets in {folder}...")
            
            # Load the datasets
            tweets = pd.read_csv(tweets_path, encoding='utf-8')
            users = pd.read_csv(users_path, encoding='utf-8')

            # Example processing: Just printing out the number of rows in each file
            print(f"Tweets: {tweets.shape[0]} rows, Users: {users.shape[0]} rows")

            # Reduce the features 
            Tweets_features = ["user_id", "retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"]
            Users_features = ["id", "statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "created_at"]
            tweets = tweets[Tweets_features]
            users = users[Users_features]

            # Convert date feature to years active
            users['created_at'] = pd.to_datetime(users['created_at'])
            users['account_age_years'] = 2015 - users['created_at'].dt.year

            # Drop Date
            users = users.drop(["created_at"], axis=1)

            # Convert all features to numeric
            users = users.apply(pd.to_numeric, errors='coerce')
            tweets = tweets.apply(pd.to_numeric, errors='coerce')

            # Missing Values
            # Fill missing values for all numeric columns in tweets DataFrame
            for col in tweets.columns:
                if pd.api.types.is_numeric_dtype(tweets[col]):
                    tweets[col] = tweets[col].fillna(0)

            # Fill missing values for all numeric columns in users DataFrame
            for col in users.columns:
                if pd.api.types.is_numeric_dtype(users[col]):
                    users[col] = users[col].fillna(0)

            # User Feature Eng
            users['followers_to_friends_ratio'] = users['followers_count'] / users['friends_count']
            users['followers_to_friends_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            users['followers_to_friends_ratio'] = users['followers_to_friends_ratio'].fillna(0)

            # Aggregate Tweet
            tweets["num_tweets"] = 1
            tweets = tweets.groupby(['user_id']).sum().reset_index()

            # Feature Eng Tweets
            tweets["retweet_ratio"] = tweets["retweet_count"]/tweets["num_tweets"]
            tweets["reply_ration"] = tweets["reply_count"]/tweets["num_tweets"]

            # Normalize
            scaler = MinMaxScaler()
            users.iloc[:,1:] = scaler.fit_transform(users.iloc[:,1:])
            tweets.iloc[:,1:] = scaler.fit_transform(tweets.iloc[:,1:])

            # Merge
            merged_df = pd.merge(tweets, users, left_on='user_id', right_on='id', how='inner')

            # Drop the 'id' column from the merged DataFrame
            merged_df = merged_df.drop('id', axis=1)

            # Add bot feature
            if folder == 'E13.csv' or folder == 'TFP.csv':
                merged_df["bot"] = 0
                users["bot"] = 0
            else:
                merged_df["bot"] = 1
                users["bot"] = 1

            # Define the new folder path
            new_folder_path = f'{base_directory + folder}/clean'

            # Create the folder if it does not exist
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            merged_df.to_csv(f'{new_folder_path}/clean_merged.csv', index=False, encoding='utf-8')
            users.to_csv(f'{new_folder_path}/clean_users.csv', index=False, encoding='utf-8')
            print("******FILES SAVED********\n\n")

class clean_cresci_2017:
    def clean_data(self, base_directory = "../Data/cresci-2017.csv/datasets_full.csv/"):
        """
        Clean data for Cresci 2017 dataset.

        Args:
            base_directory (str): Base directory containing dataset folders.

        Returns:
            None
        """
        # List all main folders in the base directory
        main_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

        # Loop through each main folder
        for main_folder in main_folders:

            # Construct the path to the nested folder, assuming the repeated structure
            nested_folder_path = os.path.join(base_directory, main_folder, main_folder)
            
            # Construct the full paths to tweets.csv and users.csv within the nested folder
            tweets_path = os.path.join(nested_folder_path, "tweets.csv")
            users_path = os.path.join(nested_folder_path, "users.csv")

            # Now you can process these datasets as needed
            print(f"Processing datasets in {main_folder}...")
            
            # Load the datasets
            tweets = pd.read_csv(tweets_path, encoding='utf-8')
            users = pd.read_csv(users_path, encoding='utf-8')

            # Example processing: Just printing out the number of rows in each file
            print(f"Tweets: {tweets.shape[0]} rows, Users: {users.shape[0]} rows")

            # Reduce the features 
            Tweets_features = ["user_id", "retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"]
            Users_features = ["id", "statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "created_at"]
            tweets = tweets[Tweets_features]
            users = users[Users_features]

            # Convert date feature to years active
            users['created_at'] = pd.to_datetime(users['created_at'])
            users['account_age_years'] = 2017 - users['created_at'].dt.year

            # Drop Date
            users = users.drop(["created_at"], axis=1)

            # Convert all features to numeric
            users = users.apply(pd.to_numeric, errors='coerce')
            tweets = tweets.apply(pd.to_numeric, errors='coerce')

            # Missing Values
            # Fill missing values for all numeric columns in tweets DataFrame
            for col in tweets.columns:
                if pd.api.types.is_numeric_dtype(tweets[col]):
                    tweets[col] = tweets[col].fillna(0)

            # Fill missing values for all numeric columns in users DataFrame
            for col in users.columns:
                if pd.api.types.is_numeric_dtype(users[col]):
                    users[col] = users[col].fillna(0)

            # User Feature Eng
            users['followers_to_friends_ratio'] = users['followers_count'] / users['friends_count']
            users['followers_to_friends_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            users['followers_to_friends_ratio'] = users['followers_to_friends_ratio'].fillna(0)

            # Aggregate Tweet
            tweets["num_tweets"] = 1
            tweets = tweets.groupby(['user_id']).sum().reset_index()

            # Feature Eng Tweets
            tweets["retweet_ratio"] = tweets["retweet_count"]/tweets["num_tweets"]
            tweets["reply_ration"] = tweets["reply_count"]/tweets["num_tweets"]

            # Normalize
            scaler = MinMaxScaler()
            users.iloc[:,1:] = scaler.fit_transform(users.iloc[:,1:])
            tweets.iloc[:,1:] = scaler.fit_transform(tweets.iloc[:,1:])

            # Merge
            merged_df = pd.merge(tweets, users, left_on='user_id', right_on='id', how='inner')

            # Add bot feature
            if main_folder == 'genuine_accounts.csv':
                merged_df["bot"] = 0
                users["bot"] = 0
            else:
                merged_df["bot"] = 1
                users["bot"] = 1

            # Define the new folder path
            new_folder_path = f'{base_directory + main_folder}/{main_folder}/clean'

            # Create the folder if it does not exist
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            merged_df.to_csv(f'{new_folder_path}/clean_merged.csv', index=False, encoding='utf-8')
            users.to_csv(f'{new_folder_path}/clean_users.csv', index=False, encoding='utf-8')
            print("******FILES SAVED********\n\n")
