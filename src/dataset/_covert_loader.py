import os
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset

from src.utils import get_base_dir
from src.dataset.utils import read_paths, string_labels

class Covert(Dataset):
    def __init__(self, xy):
        columns_of_interest = ['num_pkts', 'avg_ttl', 'median_ttl', '10_percentil_ttl',
                               '25_percentil_ttl', '75_percentil_ttl', '90_percentil_ttl', 'max_ttl',
                               'min_ttl']

        self.xy = xy
        self.x = self.xy[columns_of_interest]
        self.x = torch.tensor(self.x.to_numpy()).float()
        self.y = torch.tensor(self.y["attack"].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)


def load_covert_dataset():
    paths_raw_files = [f"{get_base_dir()}/dataset/stego/PreprocessedStegoDataset-TRAINING.csv",
                       f"{get_base_dir()}/dataset/stego/PreprocessedStegoDataset-TEST.csv"]
    path_processed_dataset = f"{get_base_dir()}/datasets/stego/processed_dataset.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        dataframes = read_paths(paths_raw_files)
        dataframe = pd.concat(dataframes)
        dataset = Covert(dataframe)
        with (open(path_processed_dataset, "wb")) as f:
            pickle.dump(dataset, f)

        return dataset