import torch
import random
import numpy as np


def get_base_dir():
    return "/home/jovyan/StealthAttackDetection"


def set_reproducibility(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False


def compute_scale_pos_weight(y_train_np):
    n_pos = (y_train_np == 1).sum()
    n_neg = (y_train_np == 0).sum()
    return n_neg / n_pos


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise Exception("Boolean value expected.")
