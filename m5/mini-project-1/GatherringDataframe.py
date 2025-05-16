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


# Defining a function
def main():
    # communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # number of the process running the code
    size = comm.Get_size()  # total number of processes running
    slice_for_each_worker = int(
        Decimal(df.shape[0] / size).quantize(Decimal("1."), rounding=ROUND_HALF_UP)
    )  # Number of elements in a array for each rank
    # Creating a sender buffer array
    if rank < size - 1:
        sendbuf = df[slice_for_each_worker * rank : slice_for_each_worker * (rank + 1)]
    else:
        sendbuf = df[slice_for_each_worker * rank :]
    # Printing the result
    print("Rank: ", rank, ", sendbuf: ", sendbuf)
    recvbuf = None
    # Gathering the Information
    recvbuf = comm.gather(sendbuf, root=0)
    # Display the result
    if rank == 0:
        print("Rank: ", rank, ", recvbuf received: ", recvbuf)


# Calling a function
main()
