import pandas as pd
import torch
import os
import pickle

from typing import Optional
from torch.utils.data import Dataset
from src.utils import get_base_dir
from src.dataset.utils import remove_collinear_features, normalize_values, read_paths, string_labels, remove_labels


class Cicids2017(Dataset):

    def __init__(self, xy: pd.DataFrame, preprocess_data: Optional[bool] = False):
        if preprocess_data:
            self.xy = xy.drop([" Destination Port"], axis="columns", inplace=True)
            self.xy = normalize_values(xy)
            self.xy = remove_collinear_features(xy, 0.95)

        else:
            self.xy = xy

        self.x = torch.tensor(self.xy.to_numpy()).float()
        self.x = self.x[:, range(0, 54)]
        self.y = torch.tensor(self.xy[[" Label"]].to_numpy()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)



def _class_string2int(dataFrames: list[pd.DataFrame], labels: list[str]) -> list[pd.DataFrame]:
    ret = []
    for df in dataFrames:
        for string in labels:
            if string == "DDoS" or string == "dos" or string == "HTTPFlood":
                df = df.replace(string, 1)

            elif string == "DoS Slowhttptest" or string == "DoS slowloris" or string == "slowite" or string == "SlowrateDoS":
                df = df.replace(string, 2)

            else:
                df = df.replace(string, 0)

        ret.append(df)

    return ret


def load_slowdos_dataset() -> Cicids2017:
    paths_raw_files = [f"{get_base_dir()}/datasets/cicids/Wednesday-workingHours.pcap_ISCX.csv", f"{get_base_dir()}/cicids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    path_processed_dataset = f"{get_base_dir()}/datasets/cicids/processed_dataset.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        dataframes = read_paths(paths_raw_files)
        labels = string_labels(dataframes)
        dataframes = remove_labels(dataframes, labels, ["DDoS", "DoS Slowhttptest", "DoS slowloris", "BENIGN"])
        dataframes = _class_string2int(dataframes, labels)
        dataframe = pd.concat(dataframes)
        dataset = Cicids2017(dataframe, True)
        with (open(path_processed_dataset, "wb")) as file:
            pickle.dump(path_processed_dataset, dataset)

        return dataset
