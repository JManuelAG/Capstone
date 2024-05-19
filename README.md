# Capstone
https://github.com/JManuelAG/Capstone/tree/main

# Twitter Bot Detection 

Welcome to the TwiTwitter Bot Detection! This project provides a comprehensive framework for data import, preprocessing, modelling, and evaluation using Python. It includes various modules for different stages of the machine learning pipeline, such as importing data, cleaning, splitting, modelling, and evaluating machine learning models.

## What the Project Does

This project offers the following functionalities:

- **Data Import and Preprocessing:**
  - *Importing Data:* The project provides utilities to import data from CSV files into pandas DataFrames.
  - *Cleaning and Preprocessing:* There are modules to preprocess and clean datasets, including handling missing values, encoding categorical variables, and scaling numerical features.

- **Data Splitting:**
  - The project allows for splitting datasets into training, testing, and validation sets with customizable proportions.

- **Modeling:**
  - Users can train various machine learning models using scikit-learn, such as K-Nearest Neighbors, Random Forest, and Logistic Regression.
  - Hyperparameter tuning and cross-validation are supported for optimizing model performance.

- **Evaluation:**
  - The project provides tools for evaluating model performance using a range of metrics, including accuracy, precision, recall, F1 score, AUC-ROC, and Matthews correlation coefficient.
  - Visualizations such as confusion matrices and ROC curves are generated to aid in understanding model behaviour.

## Why the Project is Useful

This project is useful for:
- Data analysts and data scientists looking to streamline their machine learning workflow.
- Researchers and practitioners in academia and industry who require a standardized framework for building and evaluating machine learning models.
- Students learning about data science and machine learning, as it provides hands-on experience with common tasks and techniques.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the Repository:** Clone the repository to your local machine using `git clone`.
2. **Set Up Python Environment:** Ensure you have Python installed (preferably Python 3.6 or later) along with the necessary libraries listed in the `requirements.txt` file.
3. **Explore the Notebooks:** Explore the notebooks in the `Code_Testing` directory to understand the project workflow and functionalities.
4. **Customize and Run the Code:** Customize the code and parameters as needed for your specific dataset and analysis requirements. Run the notebooks sequentially, executing each cell to observe the results and analysis.

## Where to Get Help

If you need help or have any questions about the project, you can:
- Refer to the comments and documentation within the code for detailed explanations of each functionality.
- Reach out to the project maintainer or contributors for assistance.
- Seek help from relevant online communities or forums specializing in data science and machine learning.

## Maintainers and Contributors

This project was developed and maintained by Group D. Contributions from other contributors are welcome and appreciated. Feel free to submit bug reports, feature requests, or pull requests to improve the project.

## Dependencies

This project relies on the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Ensure that these libraries are installed in your Python environment to run the code successfully. You can install them via pip using the command `pip install -r requirements.txt`.

## Description
This repository contains code for a bot detection system designed for analyzing Twitter data. The system includes modules for data preprocessing, feature selection, model testing, and evaluation.

## Important Files and Subdirectories

### 1. `cresci_2015` Directory

- `cresci_2015.csv`: The Cresci 2015 dataset containing Twitter data for bot detection research.

### 2. `Code` Directory

#### a. `clean_cresci_2015.py`

- **Description**: This module provides functionality for cleaning and preprocessing the Cresci 2015 dataset.
- **Methods**:
  - `load_data()`: Loads the Cresci 2015 dataset.
  - `preprocess_data()`: Performs preprocessing steps on the dataset.
  - `explore_data()`: Provides descriptive statistics and visualizations.
  - `split_train_test()`: Splits the dataset into training and testing sets.

#### b. `evaluate.py`

- **Description**: This module implements methods for evaluating classification models.
- **Methods**:
  - `__init__(true_values, predicted_values, predicted_probabilities=None)`: Initializes evaluation instance.
  - `accuracy()`, `confusion_matrix()`, `precision()`, `recall()`, `f1()`, `auc()`, `mcc()`: Evaluation metrics.
  - `get_all_metrics()`: Returns all evaluation metrics.
  - `plot_confusion_matrix()`, `plot_roc_curve()`: Visualization methods.

#### c. `feature_selection.py`

- **Description**: This module provides feature selection methods such as correlation analysis, chi-square test, and mutual information classifier.
- **Methods**:
  - `__init__(data)`: Initializes feature selection instance.
  - `select_features(type_selection)`: Selects features based on the specified method.
  - `correlation()`, `chi2()`, `mutual_classifier()`: Feature selection methods.
  - `pair_plot(num_feat)`, `correlation_map(num_feat)`: Visualization methods.

#### d. `import_data.py`

- **Description**: This module handles importing, sampling, and splitting the dataset.
- **Methods**:
  - `read_and_sample_data()`: Reads and samples the dataset.
  - `split_dataset()`: Splits the dataset into training, testing, and validation sets.

#### e. `models_test.py`

- **Description**: This module facilitates model testing, parameter tuning, and evaluation using grid search.
- **Methods**:
  - `load_models()`, `change_model_parameters()`, `save_current_parameters()`: Model handling methods.
  - `fit_all_models()`, `grid_search()`, `predict_model()`: Model testing and evaluation methods.

#### f. `testing_environment.py`

- **Description**: This module provides an integrated testing environment for evaluating multiple models.
- **Methods**:
  - `__init__()`: Initializes testing environment instance.
  - `save_results()`: Saves evaluation results to a CSV file.
  - `run_tests()`: Runs tests for specified model configurations.

### 3. `Parameters` Directory

- Contains pre-trained models with their parameters.

### 4. `Outputs` Directory

- Contains output files such as evaluation results.

## Usage Example

1. Instantiate necessary classes and modules.
2. Use appropriate methods to preprocess data, select features, and test models.
3. Evaluate model performance and visualize results.
4. Save evaluation results for further analysis.

## Roadmap

- **Step 1**: Preprocess the dataset using `clean_cresci_2015.py`.
- **Step 2**: Select features using `feature_selection.py`.
- **Step 3**: Test and evaluate models using `models_test.py`.
- **Step 4**: Use `testing_environment.py` to conduct comprehensive testing and analysis.

