
from create_communicator import create_comm


def main():
    comm, rank, size = create_comm()
    if rank == 0:
        # Load the data
        # df = load_data(FILENAME)
        print(comm, rank, size)
        # # Clean the data
        # df = clean_data(df)
        # # Scale the data
        # df = scale_data(df)
        # # Split the data into X and Y
        # X, Y = split_data_into_X_and_Y(df)
        # # Prepare the data for workers
        # x_train_slices, y_train_slices = prepare_data_for_workers(X, Y, size, rank)


main()
