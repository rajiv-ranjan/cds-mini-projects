
from projects_functions_library import (
    load_data,
    clean_data,
    scale_data,
    split_data_into_X_and_Y,
    estimate_coefficients,
    FILENAME,
)
from utility import download_and_unzip
from create_communicator import create_comm


def main():
    comm, rank, size = create_comm()

    download_and_unzip(
        filename="PowerPlantData.csv",
        url="https://cdn.iisc.talentsprint.com/CDS/Datasets/PowerPlantData.csv",
    )

    if rank == 0:
        df = load_data(FILENAME)

        df = clean_data(df)

        df = scale_data(df)

        X, Y = split_data_into_X_and_Y(df)

        coefficients = estimate_coefficients(X, Y)

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
