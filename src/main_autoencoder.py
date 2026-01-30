import torch
import time
import os
import torch.nn as nn

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
    print(f"Training Time: {time.time() - start_time:.2f}s")

    # Testing (ora test_loader contiene x e y)
    errors, labels = trainer.test(test_loader)
    errors,labels = trainer.test_few_shot(test_loader,x_class1)

    

if __name__ == '__main__':
    main()