from mpi4py import MPI  # Importing mpi4py package from MPI module
import numpy as np
import pandas as pd
from decimal import (
    Decimal,
    ROUND_HALF_UP,
)  # Importing Decimal, ROUND_HALF_UP functions from the decimal package

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


def dividing_data(dataset, size_of_workers):
    # Divide the data among the workers
    slice_for_each_worker = int(
        Decimal(dataset.shape[0] / size_of_workers).quantize(
            Decimal("1."), rounding=ROUND_HALF_UP
        )
    )
    print("Slice of data for each worker: {}".format(slice_for_each_worker))
    data_for_worker = []
    for i in range(0, size_of_workers):
        if i < size_of_workers - 1:
            data_for_worker.append(
                dataset[slice_for_each_worker * i : slice_for_each_worker * (i + 1)]
            )
        else:
            data_for_worker.append(dataset[slice_for_each_worker * i :])
    return data_for_worker


# Defining a function
def main():
    # communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # number of the process running the code
    size = comm.Get_size()  # total number of processes running
    data = None  # Starting with an empty  data
    if rank == 0:
        # Creating a Numpy array.
        data = dividing_data(df, size)
    # scatter operation
    received_data = comm.scatter(data, root=0)
    # Displaying the result
    print("Rank: ", rank, ", recvbuf: ", received_data)


# Calling the main function
main()
