
# Importing pandas
import pandas as pd

# Importing Numpy
import numpy as np

# Importing MPI from mpi4py package
from mpi4py import MPI

# Importing sqrt function from the Math
from math import sqrt

# Importing Decimal, ROUND_HALF_UP functions from the decimal package
from decimal import Decimal, ROUND_HALF_UP
import time

import seaborn as sns
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Pastel1_4

from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# import tensorflow as tf
import nbformat


from utility import download_and_unzip
# FILENAME = "/content/PowerPlantData.csv" # File path
FILENAME = "PowerPlantData.csv"  # File path


# YOUR CODE HERE to Define a function to load the data
# Function to load the data
def load_data(file_path):
    """
    Function to load the data from the given file path
    """
    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path, header=0, sep=",", engine="python")

    # Renaming columns in a DataFrame
    data.rename(
        columns={
            "AT": "AmbientTemperature",
            "V": "ExhaustVaccum",
            "AP": "AmbientPressure",
            "RH": "RelativeHumidity",
            "PE": "EnergyOutput",
        },
        inplace=True,
    )
    # Return the DataFrame
    return data


def clean_data(df):
    """
    Function to clean the data
    """
    print("before dropping missing values: {}".format(df.shape))
    df_ = df.dropna().reset_index(drop=True)  # Drop rows with missing values
    print("after dropping missing values {}".format(df_.shape))
    df_ = df.drop_duplicates().reset_index(drop=True)
    print("after dropping duplicate values {}".format(df_.shape))
    return df_


def scale_data(df):
    """
    Function to standardize the data
    """
    # Standardizing the data
    df_ = (df - df.mean()) / df.std()
    return df_


def split_data_into_X_and_Y(df):
    """
    Function to split the data into X and Y
    """
    # Extracting the target column as Y and remaining columns as X
    target_column = "EnergyOutput"  # Replace with the name of your target column
    X = df.drop(columns=[target_column])  # Features
    Y = df[target_column]  # Target

    return X, Y


def estimate_coefficients(X, Y):
    # Calculate the coefficients using the formula
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

    return coefficients
