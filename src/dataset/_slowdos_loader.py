import pandas
import torch
import os
import pickle

from torch.utils.data import Dataset
from src.support.utils import get_base_dir
from src.dataset.utils import read_paths, string_labels, remove_labels


def _class_string2int(dataFrames: list[pandas.DataFrame], labels: list[str]) -> list[pandas.DataFrame]:
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


def load_slowdos_dataframe() -> pandas.DataFrame:
    paths_raw_files = [f"{get_base_dir()}/datasets/cicids/Wednesday-workingHours.pcap_ISCX.csv", f"{get_base_dir()}/cicids/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    path_processed_dataset = f"{get_base_dir()}/datasets/cicids/processed_dataset.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        dataframes = read_paths(paths_raw_files)
        labels = string_labels(dataframes)
        dataframes = remove_labels(dataframes, labels, ["DDoS", "DoS Slowhttptest", "DoS slowloris", "BENIGN"])
        dataframes = _class_string2int(dataframes, labels)
        dataframe = pandas.concat(dataframes)
        dataframe = dataframe.drop([" Destination Port"], axis="columns", inplace=True)
        dataframe.rename(columns={" Label": "attack"}, inplace=True)
        dataframe.to_pickle(path_processed_dataset)
        return dataframe
