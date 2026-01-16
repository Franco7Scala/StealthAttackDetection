import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for anomaly detection experiments.")
    parser.add_argument("--attack-type", type=str, choices=["slowdos", "covert", "cobalt"], default="slowdos", help="Type of attack dataset to load.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-normal-ratio", type=float, default=0.8, help="Fraction of normal data to use for training.")
    parser.add_argument("--b-max", type=int, default=0.5, help="Maximum fraction of attack data to consider for the budget.")
    parser.add_argument("--n-train-attacks", type=int, default=10, help="Number of attack samples to include in the few-shot training set.")
    parser.add_argument("--apply-normalization", type=bool, default=False, help="Enable data normalization.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the DataLoaders.")
    parser.add_argument("--z-dim", type=int, default=16, help="Latent code dimension for ARN Generator.")
    parser.add_argument("--nf-out", type=int, default=16, help="Number of neurons for ARN Generator.")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of experimental runs.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train ARN.")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Learning rate for ARN Discriminator.")
    parser.add_argument("--lr_G", type=float, default=0.001, help="Learning rate for ARN Generator.")
    parser.add_argument("--SAVE_FOLDER", type=str, default="./saved", help="Folder for saving models.")
    parser.add_argument("--nout", type=int, default=128, help="Number of neurons of the feature extractor for ARN Discriminator.")
    parser.add_argument("--nc_out", type=int, default=16,
                        help="Number of neurons of the classifier for ARN Discriminator.")
    parser.add_argument("--n_epochs_cpvae", type=int, default=16, help="Number of epochs to train CPVAE Model.")

    return parser.parse_args()
