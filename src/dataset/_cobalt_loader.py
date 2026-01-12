import pandas as pd
import torch
import os
import pickle
from typing import Iterable

from typing import Optional
from torch.utils.data import Dataset
from src.support.utils import get_base_dir
from src.dataset.utils import remove_collinear_features, normalize_values, read_paths, string_labels, remove_labels


class Cobailt(Dataset):

    def __init__(
        self,
        xy: pd.DataFrame,
        preprocess_data: bool = False,
        label_col: str = "label",
        columns_to_drop: Optional[Iterable[str]] = None
    ):
        df = xy.copy()
        if columns_to_drop is None:
            columns_to_drop = [
                'src_port',
                'ttl_values_min',
                'ttl_values_max',
                'ttl_values_mean',
                'ttl_values_mode',
                'distinct_ttl_values',
                'dst_port'
            ]

        if preprocess_data:
            df = self._drop_ob_columns(
                df,
                label_col=label_col,
                columns_to_drop=columns_to_drop
            )
            df = normalize_values(df)
            df = remove_collinear_features(df, threshold=0.95)

        self.xy = df


        self.y = torch.tensor(
            self.xy[[label_col]].to_numpy(),
            dtype=torch.float32
        )

        self.x = torch.tensor(
            self.xy.drop(columns=[label_col]).to_numpy(),
            dtype=torch.float32
        )

    def _drop_ob_columns(
        self,
        df: pd.DataFrame,
        label_col: str,
        columns_to_drop: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:

        df = df.copy()

        object_cols = df.select_dtypes(include=["object"]).columns
        object_cols = [c for c in object_cols if c != label_col]

        manual_cols = []
        if columns_to_drop is not None:
            manual_cols = [c for c in columns_to_drop if c in df.columns]

        cols_to_drop = list(set(object_cols + manual_cols))
        df.drop(columns=cols_to_drop, inplace=True)

        return df

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.xy)
    

def _class_string2int(dataFrames: list[pd.DataFrame], labels: list[str]) -> list[pd.DataFrame]:
    ret = []
    for df in dataFrames:
        for string in labels:
            if string == "Malicious":
                df = df.replace(string, 1)

            else:
                df = df.replace(string, 0)

        ret.append(df)

    return ret


def load_cobalt_dataset() -> Cobailt:
    paths_raw_files = [f"{get_base_dir()}/dataset/cobailtstrike.csv", f"{get_base_dir()}/dataset/benign.csv"]
    path_processed_dataset = f"{get_base_dir()}/datasets/ccc/processed_dataset.pkl"

    if os.path.isfile(path_processed_dataset):
        return pickle.load(open(path_processed_dataset, "rb"))

    else:
        dataframes = read_paths(paths_raw_files)
        labels = string_labels(dataframes)
        dataframes = remove_labels(dataframes, labels, ["Benign","Malicious"])
        dataframes = _class_string2int(dataframes, labels)
        dataframe = pd.concat(dataframes)
        dataset = Cobailt(dataframe, preprocess_data=True,label_col='label')
        with (open(path_processed_dataset, "wb")) as file:
            pickle.dump(dataset,file)

        return dataset