from src.dataset._slowdos_loader import load_slowdos_dataset
from src.dataset._covert_loader import load_covert_dataset
from src.dataset._cobalt_loader import load_cobalt_dataset


def load_dataset(attack_type, n_train, n_validation, n_test, device):
    if attack_type.lower() == "slowdos":
        dataset = load_slowdos_dataset()

    elif attack_type.lower() == "covert":
        dataset = load_covert_dataset()

    elif attack_type.lower() == "cobalt":
        dataset = load_cobalt_dataset()

    else:
        raise Exception(f"Unknown dataset for attack type '{attack_type}'!")

    dataset = _split_dataset(dataset, n_train, n_validation, n_test)
    return dataset.to(device)


def _split_dataset(dataset, n_train, n_validation, n_test):
    #TODO to be implemented
    pass

