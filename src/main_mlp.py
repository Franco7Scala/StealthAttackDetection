import torch
import time
import os
import torch.nn as nn

from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility, print_args
from src.support.focal_loss import FocalLoss
from src.MLP.model import SimpleMLP 
from src.MLP.trainer import SupervisedTrainer

def main():
    # 1. Setup Argomenti e Ambiente
    args = parse_arguments()
    print_args(args)
    set_reproducibility(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    # 2. Caricamento Dati
    # Per l'MLP supervisionato usiamo x_train_few_shot e y_train_few_shot
    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    
    # Otteniamo i dataloader

    _, train_few_shot_loader, test_loader = get_dataloaders(
        x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args
    )

    input_size = x_train_few_shot.shape[1]
    attack_type = args.attack_type

    # 3. Inizializzazione Modello MLP
    # nc = input,  hidden layer size, n_classes = 1 (binary classification)
    model = SimpleMLP(nc=input_size, n_classes=1).to(device)

    # 4. Optimizer & Criterion
    # Per la classificazione binaria con logits si usa BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_D)
    criterion = FocalLoss(alpha=0.5, gamma=64, reduction='mean')
    
    trainer = SupervisedTrainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device
    )

    # 5. Training Phase (Supervisionata)
    print(f"\nStarting SimpleMLP supervised training on {attack_type}...")
    start_time = time.time()
    
    # fit
    trainer.fit(train_few_shot_loader, epochs=args.num_epochs)
    
    end_time = time.time()
    print(f"Training done! Time: {end_time - start_time:.2f} seconds")

    # 6. Testing Phase
    print(f"\nStarting MLP evaluation on test set...")

    probs, labels = trainer.test(test_loader)


if __name__ == '__main__':
    main()