from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments

def main():
    args = parse_arguments()
    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    train_unsupervised_loader, train_few_shot_loader, test_loader = get_dataloaders(x_train_unsupervised,
                                                                                    x_train_few_shot, y_train_few_shot,
                                                                                    x_test, y_test, args)

