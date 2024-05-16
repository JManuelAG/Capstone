from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FeatureSelection:
    def __init__(self, data):
        """
        Initialize the feature selection object with input data.

        Args:
            data (DataFrame): Input data containing features and target variable.

        Attributes:
            data (DataFrame): Input data.
            X (DataFrame): Features.
            y (Series): Target variable.
            values (DataFrame): DataFrame containing feature importance values.
            list_values (list): List of selected feature names.
        """
        self.data = data
        self.X = data.drop("bot", axis=1)
        self.y = data["bot"]
        self.values = pd.DataFrame()
        self.list_values = []

    def select_features(self, type_selection="correlation"):
        """
        Select features based on a specified method.

        Args:
            type_selection (str): Method for feature selection. Options: "correlation", "chi2", "classifier".

        Returns:
            list: List of selected feature names.
        """
        if type_selection == "correlation":
            values, list_values = self.correlation()
            return list_values
        
        if type_selection == "chi2":
            values, list_values = self.chi2()
            return list_values
        
        if type_selection == "classifier":
            values, list_values = self.mutual_classifier()
            return list_values

    def correlation(self):
        """
        Perform correlation analysis to select features.

        Returns:
            DataFrame: DataFrame containing feature correlations.
            list: List of selected feature names.
        """
        # Correlation Analysis 
        values = pd.DataFrame(self.data.corr()["bot"].abs().sort_values(ascending=False)[1:])
        values.reset_index(inplace=True)

        # Rename the columns to 'feature' and 'correlation'
        values.columns = ['feature', 'correlation']
        self.values = values
        
        # Create a list of values
        list_values = list(values.feature)
        self.list_values = list_values

        return values, list_values
    
    def chi2(self):
        """
        Perform chi-squared test for feature selection.

        Returns:
            DataFrame: DataFrame containing feature importances.
            list: List of selected feature names.
        """
        # Initialise the SelectKBest, for this case we will use the chi2
        skb = SelectKBest(score_func=chi2, k="all")

        # Fit the SelectKBest
        skb_chi2 = skb.fit(self.X, self.y)

        # Get the features in a data frame
        feat_imp_skb_chi2 = pd.DataFrame(
            {'feature': self.X.columns[skb_chi2.get_support()].values,
            'importance': skb_chi2.scores_
            }
            )

        # sort the results in order
        values = feat_imp_skb_chi2.sort_values('importance', ascending=False)
        self.values = values

        # Create a list for the values 
        list_values = list(values.feature.values)
        self.list_values = list_values

        return values, list_values
    
    def mutual_classifier(self):
        """
        Perform mutual information for feature selection.

        Returns:
            DataFrame: DataFrame containing feature importances.
            list: List of selected feature names.
        """
        # Now we will use the mutual_info_classif
        skb = SelectKBest(score_func=mutual_info_classif, k="all")

        # Fit the SelectKBest
        skb_mutual_info = skb.fit(self.X, self.y)

        # Get the features in a data frame
        feat_imp_skb_Minf = pd.DataFrame(
            {'feature': self.X.columns[skb_mutual_info.get_support()].values,
            'importance': skb_mutual_info.scores_
            }
            )

        # See results
        values = feat_imp_skb_Minf.sort_values('importance', ascending=False)
        self.values = values

        # Create a list for the values 
        list_values = list(values.feature.values)
        self.list_values = list_values

        return values, list_values

    def pair_plot(self, num_feat="all"):
        """
        Plot pair plots for the selected features.

        Args:
            num_feat (int or str): Number of features to include in the pair plots. Default is 'all'.

        Returns:
            None

        Raises:
            ValueError: If num_feat is not 'all' or an integer between 1 and (number of features - 1).
        """
        # Check if num_feat is 'all' or a valid number within the range
        if num_feat == 'all':
            data = self.data
        else:
            # Validate num_feat if it is a numerical value
            if isinstance(num_feat, int) and num_feat > 0 and num_feat < len(self.list_values):
                # Use .loc instead of direct slicing with brackets for clarity and correctness
                data = self.data.loc[:, self.list_values[:num_feat] + ['bot']]
            else:
                raise ValueError(f"num_feat must be 'all' or an integer between 1 and {len(self.list_values) - 1}")
        
        # Plotting the pairplot with hue set to 'bot'
        sns.pairplot(data, hue='bot')
        
    def correlation_map(self, num_feat='all'):
        """
        Plot a heatmap of correlations among variables.

        Args:
            num_feat (int or str): Number of features to include in the correlation map. Default is 'all'.

        Returns:
            None

        Raises:
            ValueError: If num_feat is not 'all' or an integer between 1 and (number of features - 1).
        """
        # Check if num_feat is 'all' or a valid number within the range
        if num_feat == 'all':
            data = self.data
        else:
            # Validate num_feat if it is a numerical value
            if isinstance(num_feat, int) and num_feat > 0 and num_feat < len(self.list_values):
                # Use .loc instead of direct slicing with brackets for clarity and correctness
                data = self.data.loc[:, self.list_values[:num_feat] + ['bot']]
            else:
                raise ValueError(f"num_feat must be 'all' or an integer between 1 and {len(self.list_values) - 1}")
            
        # Correlation among variables
        plt.figure(figsize=(20,15))
        sns.heatmap(data.corr(), annot=True)

        
        