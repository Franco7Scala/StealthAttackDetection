import numpy as np
import torch
import random

def set_reproducibility(seed):
    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

def get_base_dir():
    return "../"
