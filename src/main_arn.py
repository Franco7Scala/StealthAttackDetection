import json
import os
import sys
import numpy as np
import torch

from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.arn.trainer import ARN

from src.arn.utils import load_arn_models, get_auc, get_auprc, predict
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility


def run(params, args):
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
        x_train_unsupervised,x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
        train_loader, _, test_loader = get_dataloaders(x_train_unsupervised, x_train_few_shot,y_train_few_shot,
                                                                                        x_test, y_test, args)

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

def main():
    args = parse_arguments()

    params = vars(args)
    seed = params['seed']
    set_reproducibility(seed)

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f'Device: {device}')

    os.makedirs(params['SAVE_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(params['SAVE_FOLDER'], 'models'), exist_ok=True)

    params['device'] = device
    params['seed'] = 42

    x_train_unsupervised, _, _, _, _ = load_dataset(args)

    params['show'] = False
    params['nc'] = x_train_unsupervised.shape[1]
    print(params['nc'])

    run(params, args)

if __name__ == '__main__':
    main()

