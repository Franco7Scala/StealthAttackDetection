import torch
import time
import os
import torch.nn as nn
import pandas as pd
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility, print_args
from src.Autoencoder_simple.model import SimpleAutoencoder
from src.Autoencoder_simple.AE_trainer import AE_Trainer

def main():
    args = parse_arguments()
    print_args(args)
    set_reproducibility(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Caricamento
    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)

    mask_class1 = (y_train_few_shot == 1)
    x_class1 = x_train_few_shot[mask_class1]
    
    # Otteniamo i dataloader (train_unsupervised_loader restituisce solo x)
    train_loader, _, test_loader = get_dataloaders(
        x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args
    )

    input_size = x_train_unsupervised.shape[1]

    # Modello
    model = SimpleAutoencoder(nc=input_size, n_latent=8, nout=16).to(device)

    # Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_G)
    criterion = nn.MSELoss() 

    trainer = AE_Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
  
    # Training
    print(f"\nStarting Autoencoder training on {args.attack_type}...")
    start_time = time.time()
    trainer.fit(train_loader, epochs=args.num_epochs)
    end_time = time.time()
    training_time = round((end_time - start_time) / 60, 3)
    print(f"Training Time: {end_time - start_time:.2f}s")


    # Testing (ora test_loader contiene x e y)
    errors, labels, metrics_ae = trainer.test(test_loader)
    errors,labels,metrics_aemin = trainer.test_few_shot(test_loader,x_class1)
    errors,labels,metrics_aemedian = trainer.test_few_shot_median(test_loader,x_class1)
    results = [

        {
            "run_id": args.n_runs,
            "attack_type": args.attack_type,
            "model": "AE",
            "training_time": training_time,
            **metrics_ae
        },

        {
            "run_id": args.n_runs,
            "attack_type": args.attack_type,
            "model": "AE_fewshot_min",
            "training_time": training_time,
            **metrics_aemin
        },

        {
            "run_id": args.n_runs,
            "attack_type": args.attack_type,
            "model": "AE_fewshot_median",
            "training_time": training_time,
            **metrics_aemedian
        }

    ]

    # =========================
    # OUTPUT DIR
    # =========================

    save_dir = os.path.join(
        args.SAVE_FOLDER,
        "run_autoencoder",
        args.attack_type
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"run_{args.n_runs}.csv"
    )

    pd.DataFrame(results).to_csv(save_path, index=False)

    print(f"\nSaved results to: {save_path}")
    

    

if __name__ == '__main__':
    main()