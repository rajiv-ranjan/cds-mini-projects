
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
    # print("before dropping missing values: {}".format(df.shape))
    df_ = df.dropna().reset_index(drop=True)  # Drop rows with missing values
    # print("after dropping missing values {}".format(df_.shape))
    df_ = df.drop_duplicates().reset_index(drop=True)
    # print("after dropping duplicate values {}".format(df_.shape))
    return df_


def scale_data(df):
    # Standardizing the data
    df_ = (df - df.mean()) / df.std()
    return df_


def split_data_into_X_and_Y(df):
    # Extracting the target column as Y and remaining columns as X
    target_column = "EnergyOutput"  # Replace with the name of your target column
    X = df.drop(columns=[target_column])  # Features
    Y = df[target_column]  # Target

    return X, Y


def estimate_coefficients(X, Y):
    # Calculate the coefficients using the formula
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

    return coefficients


def fit(x, y):
    # YOUR CODE HERE
    # Add a column of ones to X for the intercept term
    # print(x, x.shape)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    # print(x, x.shape)
    coefficients = estimate_coefficients(x, y)
    return coefficients


# Call the fit function


def predict(x, intercept, coefficients):
    """
    y = b_0 + b_1*x + ... + b_i*x_i
    """
    # YOUR CODE HERE
    predictions = intercept + np.dot(x, coefficients)
    return predictions


def calculate_rmse(y_true, y_pred):
    rmse_value_1 = np.sqrt(np.mean(np.square(y_true - y_pred)))
    rmse_value_1

    rmse_value_2 = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_value_2
    return rmse_value_1

from sklearn.utils import shuffle


def train_test_split(X, Y, train_size=0.7):
    X, Y = shuffle(X, Y, random_state=42)
    split_index = int(len(X) * train_size)

    # Split the data into train and test sets
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    return X_train, X_test, Y_train, Y_test
from mpi4py import MPI


# Defining a function
def create_comm():
    # Creating a Communicator
    comm = MPI.COMM_WORLD
    # number of the process running the code
    rank = comm.Get_rank()
    # total number of processes running
    size = comm.Get_size()
    # Displaying the rank and size of the communicator
    print("Rank: {}, Size: {}".format(rank, size))
    return comm, rank, size


# comm,rank,size = create_comm()


def dividing_data(x_train, y_train, size_of_workers):
    # Size of the slice
    slice_for_each_worker = int(
        Decimal(x_train.shape[0] / size_of_workers).quantize(
            Decimal("1."), rounding=ROUND_HALF_UP
        )
    )
    # print("Slice of data for each worker: {}".format(slice_for_each_worker))
    # YOUR CODE HERE
    # Dividing the data into slices
    x_train_slices = []
    y_train_slices = []
    for i in range(size_of_workers):
        start_index = i * slice_for_each_worker
        end_index = (
            (i + 1) * slice_for_each_worker
            if i != size_of_workers - 1
            else x_train.shape[0]
        )
        x_train_slices.append(x_train[start_index:end_index])
        y_train_slices.append(y_train[start_index:end_index])
    return x_train_slices, y_train_slices


def get_data_for_worker(rank, x_train_slices, y_train_slices):
    # Getting the data for the worker
    x_train = x_train_slices[rank]
    y_train = y_train_slices[rank]
    return x_train, y_train


# def get_data_for_all_workers(x_train_slices, y_train_slices):
#     # YOUR CODE HERE
#     # Getting the data for all workers
#     x_train = []
#     y_train = []
#     for i in range(len(x_train_slices)):
#         x_train.append(x_train_slices[i])
#         y_train.append(y_train_slices[i])
#     return x_train, y_train


# YOUR CODE HERE
def prepare_data_for_workers(X, Y, size_of_workers, rank, train_size=0.7):
    if rank == 0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size)
        # Preparing the data for workers
        x_train_slices, y_train_slices = dividing_data(
            X_train, Y_train, size_of_workers
        )
        # x_train, y_train = get_data_for_worker(rank, x_train_slices, y_train_slices)
    return x_train_slices, y_train_slices
