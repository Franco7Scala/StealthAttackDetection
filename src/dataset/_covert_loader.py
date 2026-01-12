import os
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset

from src.utils import get_base_dir
from src.dataset.utils import read_paths

def load_covert_dataframe():
    columns_of_interest = ['num_pkts', 'avg_ttl', 'median_ttl', '10_percentil_ttl',
                           '25_percentil_ttl', '75_percentil_ttl', '90_percentil_ttl', 'max_ttl',
                           'min_ttl', 'attack']

    paths_raw_files = [f"{get_base_dir()}/dataset/stego/PreprocessedStegoDataset-TRAINING.csv",
                       f"{get_base_dir()}/dataset/stego/PreprocessedStegoDataset-TEST.csv"]
    path_processed_dataset = f"{get_base_dir()}/datasets/stego/processed_dataset.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        dataframes = read_paths(paths_raw_files)
        dataframe = pd.concat(dataframes)
        dataframe = dataframe[columns_of_interest]
        dataframe.to_pickle(path_processed_dataset)

        return dataframe