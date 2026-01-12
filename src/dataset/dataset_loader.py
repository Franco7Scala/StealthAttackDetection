import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


from src.dataset._slowdos_loader import load_slowdos_dataset
from src.dataset._covert_loader import load_covert_dataframe
from src.dataset._cobalt_loader import load_cobalt_dataset


def load_dataset(params):
    attack_type = params.attack_type
    device = params.device

    if attack_type.lower() == "slowdos":
        dataset = load_slowdos_dataset()

    elif attack_type.lower() == "covert":
        dataframe = load_covert_dataframe()

    elif attack_type.lower() == "cobalt":
        dataset = load_cobalt_dataset()

    else:
        raise Exception(f"Unknown dataset for attack type '{attack_type}'!")

    return _split_dataframe(dataset, params)


def _split_dataframe(dataset, params):
    df_normal = dataset[dataset['attack'] == 0]
    df_attack = dataset[dataset['attack'] == 1]

    df_normal_train = df_normal.sample(frac=params.train_normal_ratio, random_state=params.seed)
    df_normal_test = df_normal.drop(df_normal_train.index)

    df_attack_budget = df_attack.sample(frac=params.b_max, random_state=params.seed)
    df_attack_test = df_attack.drop(df_attack_budget.index)

    df_attack_train = df_attack_budget[:params.n_train_attacks]

    df_test = pd.concat([df_normal_test, df_attack_test]).sample(frac=1, random_state=params.seed).reset_index(drop=True)

    x_test = df_test.drop(columns=['attack'])
    x_test = torch.tensor(x_test.to_numpy())

    y_test = df_test['attack']
    y_test = torch.tensor(y_test.to_numpy()).float()

    df_train_unsupervised = df_normal_train.copy()
    x_train_unsupervised = df_train_unsupervised.drop(columns=['attack'])
    x_train_unsupervised = torch.tensor(x_train_unsupervised.to_numpy())

    df_train_few_shot = pd.concat([df_normal_train, df_attack_train]).sample(frac=1, random_state=params.seed).reset_index(drop=True)

    x_train_few_shot = df_train_few_shot.drop(columns=['attack'])
    x_train_few_shot = torch.tensor(x_train_few_shot.to_numpy())

    y_train_few_shot = df_train_few_shot['attack']
    y_train_few_shot = torch.tensor(y_train_few_shot.to_numpy()).float()

    if params.apply_normalization:
        # TODO: Normalizzare i dati (?)
        pass

    test_dataset = Dataset(x_test, y_test)
    train_unsupervised_dataset = Dataset(x_train_unsupervised)
    train_few_shot_dataset = Dataset(x_train_few_shot, y_train_few_shot)

    return (DataLoader(train_unsupervised_dataset, batch_size=params.batch_size, shuffle=True),
            DataLoader(train_few_shot_dataset, batch_size=params.batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=params.batch_size))