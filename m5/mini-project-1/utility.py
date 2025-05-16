# This file contains the utility functions that are used in the project
import pandas as pd
import os
import subprocess
import zipfile

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

def download_if_missing(filename, url):
    """Downloads file if it doesn't exist"""
    if not os.path.exists(filename):
        try:
            subprocess.run(['wget', url], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            return False
    return False


def download_and_unzip(filename, url):
    """Downloads zip file if missing and unzips it"""
    if not download_if_missing(filename, url):
        return False
    
    if filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall()
            return True
        except zipfile.BadZipFile as e:
            print(f"Unzip failed: {e}")
            return False
    
    return True
