import pandas as pd
import numpy as np


def normalize_values(dataFrame: pd.DataFrame) -> pd.DataFrame:
    ret = dataFrame
    for column in ret.columns:
        if column not in [" Label", "target", "Attack Type"]:
            ret[column] = ret[column].replace(np.inf, 0) #replace inf with zero
            for value in ret[column].unique():
                if value < 0:
                    ret[column] = ret[column].replace(value, 0) #replace negatives with zero

            mean_value = ret[column].mean(axis=0)
            ret[column] = ret[column].replace(0, mean_value) #replace 0 with mean value of the column
            ret[column] = ret[column].fillna(mean_value) #replace na with mean value of the column

    return ret


def remove_collinear_features(dataFrame: pd.DataFrame, threshold) -> pd.DataFrame:
    ret = dataFrame
    corr_matrix = dataFrame.corr(method="pearson", min_periods=5, numeric_only=True)
    iters = range(len(corr_matrix.columns) - 1)
    drop_columns = []
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            val = abs(item.values)
            if val >= threshold:
                drop_columns.append(col.values[0])

    drop_columns = list(set(drop_columns))
    ret = ret.drop(columns=drop_columns)
    return ret
