import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class clean_cresci_2015:
    def clean_data(self, base_directory = "../Data/cresci-2015.csv/"):
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

            # Convert Data Type
            users['created_at'] = pd.to_datetime(users['created_at'])

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
            users['account_age_years'] = 2015 - users['created_at'].dt.year
            users['followers_to_friends_ratio'] = users['followers_count'] / users['friends_count']
            users['followers_to_friends_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            users['followers_to_friends_ratio'] = users['followers_to_friends_ratio'].fillna(0)

            # Drop Date
            users = users.drop(["created_at"], axis=1)

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

