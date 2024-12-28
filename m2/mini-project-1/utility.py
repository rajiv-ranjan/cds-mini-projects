# This file contains the utility functions that are used in the project
import pandas as pd

def basic_data_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns the basic details of the data like shape, info, head, describe
    """
    print(f"\n---------------------------------------------",end="\n")
    print(f"Shape of the data: {df.shape}",end="\n")
    print(f"---------------------------------------------",end="\n")

    print(f"\n---------------------------------------------",end="\n")
    print(f"Info of the data: {df.info()}",end="\n")
    print(f"---------------------------------------------",end="\n")

    print(f"\n---------------------------------------------",end="\n")
    print(f"Head of the data: {df.head()}",end="\n")
    print(f"---------------------------------------------",end="\n")

    print(f"\n---------------------------------------------",end="\n")
    print(f"Describe of the data: {df.describe()}",end="\n")
    print(f"---------------------------------------------",end="\n")
    return df
