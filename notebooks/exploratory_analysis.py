import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    """
    Loads a CSV dataset into a pandas DataFrame.
    Args:
        file_path (str): Path to the dataset file (CSV).
    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Basic statistics and info
def basic_info(df):
    """
    Prints basic information about the dataset (info, describe, and missing values).
    Args:
        df (pd.DataFrame): The dataset.
    """
    print("\n--- Data Info ---")
    print(df.info())  # Data types and non-null counts
    print("\n--- Data Description ---")
    print(df.describe())  # Summary statistics for numerical columns
    print("\n--- Missing Values ---")
    print(df.isnull().sum())  # Check for missing values

# Visualize missing values
def plot_missing_data(df):
    """
    Plots a heatmap to visualize missing values in the dataset.
    Args:
        df (pd.DataFrame): The dataset.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

# Distribution of categorical features
def plot_categorical_distributions(df, column_name):
    """
    Plots the distribution of a categorical feature.
    Args:
        df (pd.DataFrame): The dataset.
        column_name (str): The categorical column name.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column_name, data=df, palette='Set2')
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Distribution of numerical features
def plot_numerical_distributions(df, column_name):
    """
    Plots the distribution of a numerical feature.
    Args:
        df (pd.DataFrame): The dataset.
        column_name (str): The numerical column name.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column_name], kde=True, color='skyblue')
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Correlation matrix
def plot_correlation_matrix(df):
    """
    Plots a heatmap of the correlation matrix for numerical columns.
    Args:
        df (pd.DataFrame): The dataset.
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

# Pairplot to visualize relationships between numerical columns
def plot_pairplot(df, columns):
    """
    Plots a pairplot to visualize relationships between numerical columns.
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of numerical column names.
    """
    sns.pairplot(df[columns], hue='Class', palette='Set2')  # Assuming 'Class' is the target column
    plt.show()

# Main function to run the analysis
def run_analysis(file_path):
    # Load the dataset
    df = load_data(file_path)
    
    # Basic information and statistics
    basic_info(df)

    # Plot missing data
    plot_missing_data(df)

    # Example: Plot the distribution of a categorical column ('Class')
    plot_categorical_distributions(df, 'Class')

    # Example: Plot the distribution of a numerical column ('Age')
    plot_numerical_distributions(df, 'Age')

    # Plot the correlation matrix
    plot_correlation_matrix(df)

    # Example: Pairplot for numerical columns
    numerical_columns = ['Age', 'Salary', 'Experience']  # Replace with your columns
    plot_pairplot(df, numerical_columns)

# Run the analysis (replace 'your_dataset.csv' with your actual dataset path)
if __name__ == "__main__":
    file_path = 'your_dataset.csv'  # Replace with the actual dataset file path
    run_analysis(file_path)
