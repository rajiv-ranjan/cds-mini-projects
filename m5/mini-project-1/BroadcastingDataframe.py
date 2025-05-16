from mpi4py import MPI  # Importing mpi4py package from MPI module
import numpy as np
import pandas as pd

# FILENAME = "/content/iris_dataset.csv" # Storing File path
FILENAME = "iris_dataset.csv"  # Storing File path


# Defining a function to load the data
def loadData(filename):
    # Loading the dataset with column names as
    data = pd.read_csv(filename)
    # Returning the dataframe
    return data


# Calling the function loadData and storing the dataframe in a variable named df
df = loadData(FILENAME)


# Defining a function
def main():
    comm = MPI.COMM_WORLD
    id = comm.Get_rank()  # number of the process running the code
    numProcesses = comm.Get_size()  # total number of processes running
    if id == 0:
        # Generate a dictionary with arbitrary data in it
        data = df
    else:
        # start with empty data
        data = None
    # Broadcasting the data
    data = comm.bcast(data, root=0)
    # Printing the data along with the id number
    print("Rank: ", id, ", received data: ", data, "\n")


# Calling a function
main()
