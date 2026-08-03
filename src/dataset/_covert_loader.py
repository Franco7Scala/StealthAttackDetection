import os
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset

from src.support.utils import get_base_dir
from src.dataset.utils import read_paths

def load_covert_dataframe():
    columns_of_interest = ['num_pkts', 'avg_ttl', 'median_ttl', '10_percentil_ttl',
                           '25_percentil_ttl', '75_percentil_ttl', '90_percentil_ttl', 'max_ttl',
                           'min_ttl', 'num_pkts_ws', 'avg_ttl_ws', 'median_ttl_ws', '10_percentil_ttl_ws',
                           '25_percentil_ttl_ws', '75_percentil_ttl_ws', '90_percentil_ttl_ws', 'max_ttl_ws',
                           'min_ttl_ws', 'attack']

    paths_raw_files = [f"{get_base_dir()}/dataset/PreprocessedStegoDataset-TRAINING.csv",
                       f"{get_base_dir()}/dataset/PreprocessedStegoDataset-TEST.csv"]
    path_processed_dataset = f"{get_base_dir()}/dataset/processed_dataset_stego.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        window_size = 4
        dataframes = read_paths(paths_raw_files)
        dataframe = pd.concat(dataframes)
        df_extended = dataframe.rolling(window_size).mean()
        df_extended = df_extended.rename(columns={'timestamp': 'timestamp_ws',
                                                  'num_pkts': 'num_pkts_ws',
                                                  "avg_ttl": 'avg_ttl_ws',
                                                  "median_ttl": 'median_ttl_ws',
                                                  "10_percentil_ttl": "10_percentil_ttl_ws",
                                                  "25_percentil_ttl": "25_percentil_ttl_ws",
                                                  "75_percentil_ttl": "75_percentil_ttl_ws",
                                                  "90_percentil_ttl": "90_percentil_ttl_ws",
                                                  "max_ttl": "max_ttl_ws",
                                                  "min_ttl": "min_ttl_ws",
                                                  "attack": "attack_ws"})

        dataframe = pd.concat([dataframe, df_extended], axis=1)
        dataframe = dataframe.dropna()

        dataframe = dataframe[columns_of_interest]
        dataframe.to_pickle(path_processed_dataset)

        return dataframe