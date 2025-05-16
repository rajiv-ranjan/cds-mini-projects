from mpi4py import MPI  # Importing mpi4py package from MPI module
import pandas as pd
import numpy as np
# Defining a function

# FILENAME = "/content/iris_dataset.csv"  # Storing File path
FILENAME = "iris_dataset.csv"  # Storing File path


# Defining a function to load the data
def loadData(filename):
    # Loading the dataset with column names as
    data = pd.read_csv(filename)
    # Returning the dataframe
    return data


# Calling the function loadData and storing the dataframe in a variable named df
df = loadData(FILENAME)


def main():
    # Creating a communicator
    comm = MPI.COMM_WORLD
    # number of the process running the code
    rank = comm.Get_rank()
    # total number of processes running
    size = comm.Get_size()
    # master process
    if rank == 0:
        # Generate a dictionary with arbitrary data in it
        data = df
        # master process sends data to worker processes by
        # going through the ranks of all worker processes
        for i in range(1, size):
            # Sending data
            comm.send(data, dest=i, tag=i)
            # Displaying the results
            print("Process {} sent data:".format(rank), data)
    # worker processes
    else:
        # each worker process receives data from master process
        data = comm.recv(source=0, tag=rank)
        # Displaying the results
        print("Process {} received data:".format(rank), data)


# Calling the function
main()
