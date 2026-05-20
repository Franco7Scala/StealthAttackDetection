import torch
import time
import pickle
import os
import torch.nn as nn
import pandas as pd
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments
from src.model.predictive_model_noise_batch_fixed import ConcatenatedPredictiveVAE
from src.support.focal_loss import FocalLoss
from src.support.utils import set_reproducibility, print_args
from src.arn.model import Generator, Discriminator
import json

def main():
    args = parse_arguments()
    print_args(args)

    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    _, train_few_shot_loader, test_loader = get_dataloaders(x_train_unsupervised,
                                                                                    x_train_few_shot, y_train_few_shot,
                                                                                    x_test, y_test, args)

    set_reproducibility(args.seed)
    attack_type = args.attack_type
    batch_size = args.batch_size


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    input_size = x_train_unsupervised.shape[1]
    output_size = 1
    random_noise = True
    mean = 0.0
    std = 0.1
    k=2
    n_runs = args.n_runs

    name_VAE_model = f'ARN_Generator_{attack_type}_0.ckpt'
    name_MC_model = f'ARN_Discriminator_{attack_type}_0.ckpt'

    path_VAE_model = os.path.join(args.SAVE_FOLDER, 'models', name_VAE_model)
    path_MC_model = os.path.join(args.SAVE_FOLDER, 'models', name_MC_model)

    MC_model = Discriminator(nc = input_size, nc_out=args.nc_out, nout=args.nout).to(device)
    MC_model.load_state_dict(torch.load(path_MC_model))

    if args.apply_normalization:
      VAE_model = Generator(nf_in=input_size, nf_out=args.nf_out,
                           z_dim=args.z_dim, out_activation=nn.ReLU).to(device)
    else:
        VAE_model = Generator(nf_in=input_size, nf_out=args.nf_out, z_dim=args.z_dim).to(device)

    VAE_model.load_state_dict(torch.load(path_VAE_model))

    CPVAE_model = ConcatenatedPredictiveVAE(MC_model, VAE_model, (args.z_dim + args.nc_out + input_size), output_size, device,params=vars(args),
                                            random_noise=random_noise, mean=mean, std=std)
    CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.0001)
    CPVAE_criterion = nn.BCEWithLogitsLoss()
    #CPVAE_criterion = FocalLoss(gamma=64, alpha=0.5, reduction="mean")

    our_model_folder = os.path.join(args.SAVE_FOLDER, 'our_models')
    os.makedirs(our_model_folder, exist_ok=True)
    last_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'last_our_models_{args.attack_type}_{args.n_exps}_{n_runs}.pt')
    best_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'best_our_models_{args.attack_type}_{args.n_exps}_{n_runs}.pt')

    print(f"Starting {attack_type} ConcatenatedPredictiveVAE model training...")
    start = time.time()
    # -----CPVAE model training-----#
    CPVAE_model.fit(args.n_epochs_cpvae, CPVAE_optimizer, CPVAE_criterion, train_few_shot_loader,batch_size,k=k,
                    best_model_path=best_model_path, last_model_path=last_model_path)
    # -----CPVAE model training-----#
    end = time.time()
    training_time_min = (end - start) / 60
    print("ConcatenatedPredictiveVAE done!")
    print(f"Training time: {end - start:.2f} seconds")

    print('Evaluate with Last Model')

    CPVAE_model.load_state_dict(torch.load(last_model_path))
    print(f"Starting ConcatenatedPredictiveVAE testing on train set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = CPVAE_model.evaluate(train_few_shot_loader, CPVAE_criterion,
                                                                            evaluation_on="train")
    print("ConcatenatedPredictiveVAE test results:")
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}, pr_auc: {pr_auc}, gmean_macro: {gmean_macro}, Confusion Mat: {cm}, FAR: {fpr}")
    print(cr)

    print("-" * 100)

    print(f"Starting ConcatenatedPredictiveVAE testing on test set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = CPVAE_model.evaluate(test_loader, CPVAE_criterion,
                                                                            evaluation_on="test")
    print("ConcatenatedPredictiveVAE test results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}\npr_auc: {pr_auc} \n gmean_macro: {gmean_macro} \n Confusion Mat: {cm} \n FAR: {fpr}")
    print(cr)

    row = {
    "run_id": args.n_runs,   # oppure args.run_id se lo usi così
    "attack_type": args.attack_type,
    "model": "CPVAE",

    "accuracy_last": accuracy,
    "precision_last": precision,
    "recall_last": recall,
    "f1_last": f1,
    "auc_last": auc,
    "pr_auc_last": pr_auc,
    "gmean_macro_last": gmean_macro,
    "fpr_last": fpr,

    "training_time": round(training_time_min, 3)
}

    results_dir = os.path.join(args.SAVE_FOLDER, f"run_cpvae_{args.n_exps}", args.attack_type)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame([row])

    save_path = os.path.join(results_dir, f"run_last_model_{args.n_runs}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved results to: {save_path}")

    print('Evaluate with Best Model')

    CPVAE_model.load_state_dict(torch.load(best_model_path))
    print(f"Starting ConcatenatedPredictiveVAE testing on train set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro, cm, fpr = CPVAE_model.evaluate(train_few_shot_loader,
                                                                                                  CPVAE_criterion,
                                                                                                  evaluation_on="train")
    print("ConcatenatedPredictiveVAE test results:")
    print(
        f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}, pr_auc: {pr_auc}, gmean_macro: {gmean_macro}, Confusion Mat: {cm}, FAR: {fpr}")
    print(cr)

    print("-" * 100)

    print(f"Starting ConcatenatedPredictiveVAE testing on test set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro, cm, fpr = CPVAE_model.evaluate(test_loader,
                                                                                                  CPVAE_criterion,
                                                                                                  evaluation_on="test")
    print("ConcatenatedPredictiveVAE test results:")
    print(
        f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}\npr_auc: {pr_auc} \n gmean_macro: {gmean_macro} \n Confusion Mat: {cm} \n FAR: {fpr}")
    print(cr)

    row = {
    "run_id": args.n_runs,   # oppure args.run_id se lo usi così
    "attack_type": args.attack_type,
    "model": "CPVAE",

    "accuracy_best": accuracy,
    "precision_best": precision,
    "recall_best": recall,
    "f1_best": f1,
    "auc_best": auc,
    "pr_auc_last": pr_auc,
    "gmean_macro_best": gmean_macro,
    "fpr_best": fpr,

    "training_time": round(training_time_min, 3)
}

    results_dir = os.path.join(args.SAVE_FOLDER, f"run_cpvae_{args.n_exps}", args.attack_type)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame([row])

    save_path = os.path.join(results_dir, f"run_best_model_{args.n_runs}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved results to: {save_path}")

if __name__ == '__main__':
    main()
