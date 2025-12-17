
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def dataset_split_by_category(df, category_column):
    """
    Splits the dataset into multiple DataFrames based on unique values in the specified category column.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be split.
    category_column (str): The column name containing categorical values to split by.

    Returns:
    dict: A dictionary where keys are unique category values and values are corresponding DataFrames.
    """         
    unique_categories = df[category_column].unique()
    split_dfs = {category: df[df[category_column] == category] for category in unique_categories}
    return split_dfs

def replace_by_mean(df, column_name):
    """
    Replaces NaN values in a DataFrame column with the column's mean.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to fill.
    """
    mean_value = round(df[column_name].mean(),0)
    df[column_name].fillna(mean_value, inplace = True)

def show_non_numeric_values(df, column_name):
    # Print all non-numeric values in required column
    nan_values = df[column_name].isna().sum()
    non_numeric_df = df[pd.to_numeric(df[column_name], errors='coerce').isna()]
    print(f"NAN values count: {nan_values}")
    print("\nNon-numeric values:")
    print(non_numeric_df[column_name].unique())
    print("\nFirst 20 rows with non-numeric column:")
    print(non_numeric_df[column_name].head(20).tolist())


def dataset_scaler(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df

