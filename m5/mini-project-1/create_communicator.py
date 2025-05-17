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
