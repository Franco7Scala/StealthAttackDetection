import torch
import time
import pickle
import os
import torch.nn as nn

from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments
from src.Ablation_4.model_ablation_4 import model
from src.support.focal_loss import FocalLoss
from src.support.utils import set_reproducibility, print_args
from src.arn.model import Discriminator


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
    std = 0.05


 
    name_MC_model = f'ARN_Discriminator_{attack_type}_0.ckpt'


    path_MC_model = os.path.join(args.SAVE_FOLDER, 'models', name_MC_model)

    MC_model = Discriminator(nc = input_size, nc_out=args.nc_out, nout=args.nout).to(device)
    MC_model.load_state_dict(torch.load(path_MC_model))



    CPVAE_model = model(MC_model, args.nc_out, output_size, device,params=vars(args),
                                            random_noise=random_noise, mean=mean, std=std)
    CPVAE_optimizer = torch.optim.Adam(CPVAE_model.parameters(), lr=0.0001)
    CPVAE_criterion = nn.BCEWithLogitsLoss()
    #CPVAE_criterion = FocalLoss(gamma=64, alpha=0.5, reduction="mean")
    our_model_folder = os.path.join(args.SAVE_FOLDER, 'our_models')
    os.makedirs(our_model_folder, exist_ok=True)
    last_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'last_our_models_{args.attack_type}_{args.n_exps}.pt')
    best_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'best_our_models_{args.attack_type}_{args.n_exps}.pt')


    print(f"Starting {attack_type} ConcatenatedPredictiveVAE model training...")
    start = time.time()
    # -----CPVAE model training-----#
    CPVAE_model.fit(args.n_epochs_cpvae, CPVAE_optimizer, CPVAE_criterion, train_few_shot_loader,batch_size,best_model_path=best_model_path, last_model_path=last_model_path)
    # -----CPVAE model training-----#
    end = time.time()

    print("ConcatenatedPredictiveVAE done!")
    print(f"Training time: {end - start:.2f} seconds")

    #print(f"Starting ConcatenatedPredictiveVAE testing on train set...")
    #accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = CPVAE_model.evaluate(train_few_shot_loader, CPVAE_criterion,
    #                                                                        evaluation_on="train")
    #print("ConcatenatedPredictiveVAE test results:")
    #print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}, pr_auc: {pr_auc}, gmean_macro: {gmean_macro}, Confusion Mat: {cm}, FAR: {fpr}")
    #print(cr)
    print("EVALUATE LAST MODEL")
    CPVAE_model.load_state_dict(torch.load(last_model_path))
    print("-" * 100)

    print(f"Starting ConcatenatedPredictiveVAE testing on test set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = CPVAE_model.evaluate(test_loader, CPVAE_criterion,
                                                                            evaluation_on="test")
    print("ConcatenatedPredictiveVAE test results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}\npr_auc: {pr_auc} \n gmean_macro: {gmean_macro} \n Confusion Mat: {cm} \n FAR: {fpr}")
    print(cr)

    print('EVALUATE BEST MODEL')

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

if __name__ == '__main__':
    main()
