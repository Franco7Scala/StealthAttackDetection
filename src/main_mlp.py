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
import pandas as pd
def main():
    # 1. Setup Argomenti e Ambiente
    args = parse_arguments()
    print_args(args)
    set_reproducibility(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")
    base_dir = args.SAVE_FOLDER+"/"+"run_mlp"
    attack_dir = os.path.join(base_dir, args.attack_type)
    os.makedirs(attack_dir, exist_ok=True)

    run_id = args.n_runs

    # 2. Caricamento Dati
    # Per l'MLP supervisionato usiamo x_train_few_shot e y_train_few_shot
    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    
    # Otteniamo i dataloader

    _, train_few_shot_loader, test_loader = get_dataloaders(
        x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args
    )

    input_size = x_train_few_shot.shape[1]
    attack_type = args.attack_type

    y_np = y_train_few_shot.numpy()

    num_pos = (y_np == 1).sum()
    num_neg = (y_np == 0).sum()
    
    pos_weight = torch.tensor(
        num_neg / num_pos,
        dtype=torch.float,
        device=device
    )
    # 3. Inizializzazione Modello MLP
    # nc = input,  hidden layer size, n_classes = 1 (binary classification)
    model = SimpleMLP(nc=input_size, n_classes=1).to(device)

    # 4. Optimizer & Criterion
    # Per la classificazione binaria con logits si usa BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_D)
    criterion = FocalLoss(alpha=2, gamma=2, reduction='mean')
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    
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
    training_time = (end_time-start_time)/60
    # 6. Testing Phase
    print(f"\nStarting MLP evaluation on test set...")
    metrics = trainer.test(test_loader)
    #probs, labels = trainer.test(test_loader)
    row = {
        "run_id": run_id,
        "attack_type": args.attack_type,
        "model": "mlp",
        "training_time": round(training_time, 3),
        **metrics
    }

    df = pd.DataFrame([row])
    save_path = os.path.join(attack_dir, f"run_{run_id}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved to: {save_path}")


if __name__ == '__main__':
    main()