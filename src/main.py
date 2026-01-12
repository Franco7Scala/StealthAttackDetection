from src.dataset.dataset_loader import load_dataset
from src.support.arguments import parse_arguments


args = parse_arguments()
load_dataset(args)
