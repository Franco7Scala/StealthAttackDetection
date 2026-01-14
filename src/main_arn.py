import json
import os
import sys
import numpy as np
import torch

from src.dataset.dataset_loader import load_dataset
from src.arn.trainer import ARN

from src.arn.utils import load_arn_models, get_auc, get_auprc, predict

def run(params):
    attack_type = params['attack_type']

    n_runs = params['n_runs']

    if 'start_runs'  in params:
        start_runs = params['start_runs']
    else:
        start_runs = 0

    seed = params['seed']

    auc_list = []
    auprc_list = []

    for i in range(start_runs, n_runs):
        print(f'Iteration: {i}')
        params['seed'] = seed*(i+1)
        train_loader, _, test_loader = load_dataset(params)

        model = ARN(params)

        name_G = f'ARN_Generator_{attack_type}_{i}.ckpt'
        name_D = f'ARN_Discriminator_{attack_type}_{i}.ckpt'

        path_G = os.path.join(params['SAVE_FOLDER'], 'models', name_G)
        path_D = os.path.join(params['SAVE_FOLDER'], 'models', name_D)

        ### Training ###

        _ = model.train(train_loader, path_G, path_D, batch_size=params['batch_size'],
                        num_epochs=params['num_epochs'])

        ### Evaluation ###
        load_arn_models(model.G, model.D, path_G, path_D)
        y_true, y_pred = predict(model.D, params['device'], test_loader)

        auc_score = get_auc(y_true, y_pred)
        auprc_score = get_auprc(y_true, y_pred)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)
    print('AUC', auc_list, 'AUPRC', auprc_list)

def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f'Device: {device}')

    os.makedirs(params['SAVE_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(params['SAVE_FOLDER'], 'models'), exist_ok=True)

    params['device'] = device
    params['seed'] = 42

    train_loader, _, _ = load_dataset(params)

    params['show'] = False
    params['nc'] = train_loader.dataset.shape[1]
    print(params['nc'])

    run(params)

if __name__ == '__main__':
    main(sys.argv[1])

