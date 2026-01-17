import pandas as pd
import torch
import os
import pickle
from typing import Iterable

from typing import Optional
from src.support.utils import get_base_dir
from src.dataset.utils import remove_collinear_features, normalize_values, read_paths, string_labels, remove_labels
from src.dataset.utils import read_paths, string_labels, remove_labels
from src.dataset.utils import normalize_values, remove_collinear_features


def _class_string2int(
    dataframes: list[pd.DataFrame],
    labels: list[str]
) -> list[pd.DataFrame]:

    ret = []
    for df in dataframes:
        for string in labels:
            if string == "Malicious":
                df = df.replace(string, 1)
            else:
                df = df.replace(string, 0)

        ret.append(df)

    return ret


def _preprocess_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    columns_to_drop: Optional[Iterable[str]] = None
) -> pd.DataFrame:

    if columns_to_drop is None:
        columns_to_drop = [
            'src_port',
            'ttl_values_min',
            'ttl_values_max',
            'ttl_values_mean',
            'ttl_values_mode',
            'ttl_values_median',
            'distinct_ttl_values',
            'dst_port'
        ]

    df = df.copy()

    object_cols = df.select_dtypes(include=["object"]).columns
    object_cols = [c for c in object_cols if c != label_col]

    manual_cols = [c for c in columns_to_drop if c in df.columns]

    df.drop(columns=list(set(object_cols + manual_cols)), inplace=True)

    #df = normalize_values(df)
    df = remove_collinear_features(df, threshold=0.95)

    return df



def load_cobalt_dataframe(
    preprocess_data: bool = True,
    label_col: str = "label"
) -> pd.DataFrame:

    paths_raw_files = [
        f"{get_base_dir()}/dataset/cobaltstrike.csv",
        f"{get_base_dir()}/dataset/benign.csv"
    ]

    path_processed_dataset = f"{get_base_dir()}/dataset/processed_cobalt.pkl"

    if os.path.isfile(path_processed_dataset):
        return pd.read_pickle(path_processed_dataset)

    dataframes = read_paths(paths_raw_files)
    labels = string_labels(dataframes)
    dataframes = remove_labels(dataframes, labels, ["Benign", "Malicious"])
    dataframes = _class_string2int(dataframes, labels)

    df = pd.concat(dataframes, ignore_index=True)

    if preprocess_data:
        df = _preprocess_dataframe(df, label_col=label_col)

    df.rename(columns={label_col: "attack"}, inplace=True)

    df.to_pickle(path_processed_dataset)
    return df
