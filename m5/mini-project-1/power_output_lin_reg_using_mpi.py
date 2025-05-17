

from utility import download_and_unzip
from mpi_support_script import (
    load_data,
    clean_data,
    scale_data,
    split_data_into_X_and_Y,
    estimate_coefficients,
    fit,
    FILENAME,
    predict,
    calculate_rmse,
    create_comm,
    dividing_data,
    prepare_data_for_workers,
    get_data_for_worker,
    train_test_split,
)
import numpy as np
from mpi4py import MPI


def main():
    comm, rank, size = create_comm()

    download_and_unzip(
        filename="PowerPlantData.csv",
        url="https://cdn.iisc.talentsprint.com/CDS/Datasets/PowerPlantData.csv",
    )

    # x_train_slices = [None] * size
    # y_train_slices = [None] * size

    # # receive_x_train = np.empty(100000, dtype="d")
    # # receive_y_train = np.empty(100000, dtype="d")
    # receive_x_train = [None] * 100000
    # receive_y_train = [None] * 100000

    # receive_y_train = np.empty(100000, dtype="d")
    slice_shapes = None

    if rank == 0:
        df = load_data(FILENAME)

        df = clean_data(df)

        df = scale_data(df)

        X, Y = split_data_into_X_and_Y(df)

        coefficients = fit(X, Y)
        c = coefficients[0]
        m = coefficients[1:]

        x_train_slices, y_train_slices = prepare_data_for_workers(X, Y, size, rank)

        x_train_slices = [df_slice.to_numpy() for df_slice in x_train_slices]
        y_train_slices = [y_slice.to_numpy() for y_slice in y_train_slices]

        # Find the shape of each slice (they should be equal except maybe the last)
        slice_shapes = [arr.shape for arr in x_train_slices]
        print("slide shape: {slice_shapes}")  # For debugging

        print(type(x_train_slices))
        print(type(y_train_slices))
    else:
        # Initialize empty arrays for x_train_slices and y_train_slices
        x_train_slices = None
        y_train_slices = None

    # For 2 processes, e.g. shape might be [(3334, 4), (3333, 4)]
    # Let's assume all but the last are the same length: slice_len = x_train_slices[0].shape[0]
    # slice_len = x_train_slices[0].shape[0]
    # n_features = x_train_slices[0].shape[1]

    slice_len = 100000
    n_features = 100000

    # Prepare receive buffers
    receive_x_train = np.empty((slice_len, n_features), dtype=np.float64)
    receive_y_train = np.empty((slice_len,), dtype=np.float64)

    # Scatter using sendbuf and recvbuf

    comm.Scatter(x_train_slices, receive_x_train, root=0)
    comm.Scatter(y_train_slices, receive_y_train, root=0)

    # comm.Scatter(x_train_slices, receive_x_train, root=0)
    # comm.Scatter(y_train_slices, receive_y_train, root=0)

    print("RANK:{rank} shape of receive_x_train: {receive_x_train.shape}")
    print("RANK:{rank} shape of receive_y_train: {receive_y_train.shape}")

    # Scatter the data to all workers
    # x_train, y_train = scatter_data(comm, x_train_slices, y_train_slices, numDataPerRank)
    # Gather the data from all workers
    # x_train_slices, y_train_slices = gather_data(comm, x_train, y_train)

    # predicted_value = predict(X, c, m)

    # calculate_rmse(Y, predicted_value)

    # Load the data
    # df = load_data(FILENAME)
    # print(comm, rank, size)
    # # Clean the data
    # df = clean_data(df)
    # # Scale the data
    # df = scale_data(df)
    # # Split the data into X and Y
    # X, Y = split_data_into_X_and_Y(df)
    # # Prepare the data for workers
    # x_train_slices, y_train_slices = prepare_data_for_workers(X, Y, size, rank)


main()
