#!/usr/bin/env python
from pathlib import Path
from typing import Tuple
import warnings
import numpy as np
import os
import glob
import pandas as pd
import torch
import gcsfs
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

#------------- globally accessing the dataset
FILE_SYSTEM = gcsfs.core.GCSFileSystem(requester_pays=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available
forcing_path = "/Users/mpgrad/Desktop/PHD_Research/camels-20240903T2000Z"

CAMELS_ROOT = Path(forcing_path+"/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2")#specifies the path of the CAMELS
def load_forcing(basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific basin.

    :param basin: 8-digit code of basin as string.
    :return: pd.DataFrame containing the meteorological forcing data and the area of the basin as integer.
    """
    # Root directory of meteorological forcings
    forcing_path = os.path.join(CAMELS_ROOT, 'basin_mean_forcing', 'daymet')

    # path of forcing file
    files = list(glob.glob(os.path.join(forcing_path, '**', f'{basin}_*.txt'), recursive=True))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file found for Basin {basin}')
    else:
        file_path = files[0]

    # converting date to datetime index and reading it
    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep='\s+', header=3)
    dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # Load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area

def load_discharge(basin: str, area: int) -> pd.Series:
    """Load the discharge time series for a specific basin.

    :param basin: 8-digit code of basin as string.
    :param area: int, area of the catchment in square meters
    :return: A pd.Series containing the catchment normalized discharge.
    """
    # Root directory of the streamflow data
    discharge_path = os.path.join(CAMELS_ROOT, 'usgs_streamflow')


    files = list(glob.glob(os.path.join(discharge_path, '**', f'{basin}_*.txt'), recursive=True))
    if len(files) == 0:
        raise RuntimeError(f'No discharge file found for Basin {basin}')
    else:
        file_path = files[0]

    # Read-in data and converting date to datetime index
    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    with open(file_path, 'r') as fp:
        df = pd.read_csv(fp, sep='\s+', header=None, names=col_names)

    # Converting date columns to datetime index
    dates = df.Year.astype(str) + "/" + df.Mnth.astype(str) + "/" + df.Day.astype(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # Normalizing discharge from cubic feet per second to mm per day
    df['QObs'] = 28316846.592 * df['QObs'] * 86400 / (area * 10 ** 6)

    return df['QObs']

