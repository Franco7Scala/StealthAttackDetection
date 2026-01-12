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
    return parser.parse_args()
