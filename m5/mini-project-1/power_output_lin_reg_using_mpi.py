

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
    rmse_value_1 = 0.26752544372182074
    comm, rank, size = create_comm()

    download_and_unzip(
        filename="PowerPlantData.csv",
        url="https://cdn.iisc.talentsprint.com/CDS/Datasets/PowerPlantData.csv",
    )

    slice_shapes = None
    receive_x_train = None
    receive_y_train = None
    x_train_slices = []
    y_train_slices = []

    if rank == 0:
        df = load_data(FILENAME)

        df = clean_data(df)

        df = scale_data(df)

        X, Y = split_data_into_X_and_Y(df)

        x_train_slices, y_train_slices = prepare_data_for_workers(X, Y, size, rank)

        print(
            f"type of x_train_slices is {type(x_train_slices)} and length is {len(x_train_slices)}"
        )
        print(x_train_slices)
        print(
            f"type of x_train_slices is {type(x_train_slices)} and length is {len(x_train_slices)}"
        )
        print(x_train_slices)

    receive_x_train = comm.scatter(x_train_slices, root=0)
    receive_y_train = comm.scatter(y_train_slices, root=0)

    fitted_coefficients = fit(receive_x_train, receive_y_train)

    all_coefficients = comm.gather(fitted_coefficients, root=0)

    comm.Barrier()  # Ensure all processes are synchronized

    if rank == 0:
        print(f"length of all coefficients: {len(all_coefficients)}")

        # Combine the coefficients from all processes
        combined_coefficients = np.mean(all_coefficients, axis=0)
        print("Combined Coefficients:", combined_coefficients)

        final_intercept = combined_coefficients[0]
        final_coefficients = combined_coefficients[1:]
        print("Final Intercept:", final_intercept)
        print("Final Coefficients:", final_coefficients)

        # Calculate RMSE
        Y_pred = predict(X, final_intercept, final_coefficients)
        rmse_value_2 = calculate_rmse(Y, Y_pred)
        print("RMSE value without parallel execution:", rmse_value_1)
        print("RMSE value with parallel execution:", rmse_value_2)


main()
