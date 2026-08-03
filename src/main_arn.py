import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.arn.trainer import ARN
import time
from src.arn.utils import load_arn_models, get_auc, get_auprc, predict
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility, print_args
from src.arn.plotter import plot_ARN_loss, plot_pr_curve, plot_auc_curve


def run(params, args):
    attack_type = params['attack_type']

    n_runs = params['n_runs']
    auc_dir = os.path.join(params['SAVE_FOLDER'], 'auc_arn', f'auc_arn_{attack_type}_new')
    prc_dir = os.path.join(params['SAVE_FOLDER'], 'prc_arn', f'prc_arn_{attack_type}_new')
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

        name_G = f'ARN_Generator_{attack_type}_{i}_new.ckpt'
        name_D = f'ARN_Discriminator_{attack_type}_{i}_new.ckpt'

        path_G = os.path.join(params['SAVE_FOLDER'], 'models', name_G)
        path_D = os.path.join(params['SAVE_FOLDER'], 'models', name_D)

        name_G = f'ARN_Generator_{attack_type}_{i}_best_new.ckpt'
        name_D = f'ARN_Discriminator_{attack_type}_{i}_best_new.ckpt'

        path_best_G = os.path.join(params['SAVE_FOLDER'], 'models', name_G)
        path_best_D = os.path.join(params['SAVE_FOLDER'], 'models', name_D)

        ### Training ###
        start_time = time.time()
        _ = model.train(train_loader, path_G, path_D, batch_size=params['batch_size'],
                        num_epochs=params['num_epochs'], num_q_steps=5,
                        path_best_G=path_best_G, path_best_D=path_best_D)
        end_time = time.time()
        print((end_time-start_time)/60)

        ### Evaluation ###
        print('Evaluate Best Model')
        load_arn_models(model.G, model.D, path_best_G, path_best_D)
        y_true, y_pred_prob = predict(model.D, params['device'], test_loader)

        y_pred_prob = 1 - y_pred_prob
        y_pred = (y_pred_prob > 0.5) + 0.

        auc_score = get_auc(y_true, y_pred,auc_dir)
        auprc_score = get_auprc(y_true, y_pred,prc_dir)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc_ = roc_auc_score(y_true=y_true, y_score=y_pred_prob)
        cr = classification_report(y_true, y_pred, target_names=["Benign", "Attack"])

        rc_precision, rc_recall, rc_thresholds = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(rc_recall, rc_precision)

        print(f"precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc_}, pr_auc: {pr_auc}")
        print(cr)

        print('Evaluate Last Model')
        load_arn_models(model.G, model.D, path_G, path_D)
        y_true, y_pred_prob = predict(model.D, params['device'], test_loader)

        y_pred_prob = 1 - y_pred_prob
        y_pred = (y_pred_prob > 0.5) + 0.
        # y_pred = 1-y_pred

        auc_score = get_auc(y_true, y_pred_prob, auc_dir)
        auprc_score = get_auprc(y_true, y_pred_prob, prc_dir)

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc_ = roc_auc_score(y_true=y_true, y_score=y_pred_prob)
        cr = classification_report(y_true, y_pred, target_names=["Benign", "Attack"])

        rc_precision, rc_recall, rc_thresholds = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(rc_recall, rc_precision)

        print(f"precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc_}, pr_auc: {pr_auc}")
        print(cr)
     

    print('AUC', auc_list, 'AUPRC', auprc_list)
    

def main():
    args = parse_arguments()
    print_args(args)

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

