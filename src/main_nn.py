from src.dataset.dataset_loader import load_dataset, get_dataloaders
import torch
from neural_net.model import SimpleDiscriminator
from neural_net.train import Trainer
from src.support.focal_loss import FocalLoss
import argparse
import torch
import os
from src.support.arguments import parse_arguments
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import numpy as np



def run(params, args):
    n_runs = params['n_runs']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    seed = params['seed']
    device = params['device']

    for i in range(n_runs):
        print(f'\n=== Iteration {i} ===')
        current_seed = seed

        torch.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ----------------------------
        # Carica dataset
        # ----------------------------
        x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
        mask_class0 = (y_train_few_shot == 0)
        mask_class1 = (y_train_few_shot == 1)
        
       # x_class0 = x_train_few_shot[mask_class0][:1000]  
       # y_class0 = y_train_few_shot[mask_class0][:1000]

        x_class0 = x_train_few_shot[mask_class0]  
        y_class0 = y_train_few_shot[mask_class0]
        
        x_class1 = x_train_few_shot[mask_class1]
        y_class1 = y_train_few_shot[mask_class1]
        
        # ricombina
        #x_train_few_shot = torch.cat([x_class0, x_class1], dim=0)
        #y_train_few_shot = torch.cat([y_class0, y_class1], dim=0)
        
        # mescola i dati
       # perm = torch.randperm(len(y_train_few_shot))
        #x_train_few_shot = x_train_few_shot[perm]
        #y_train_few_shot = y_train_few_shot[perm]

        
        num_pos = (y_train_few_shot == 1).sum().item()
        num_neg = (y_train_few_shot == 0).sum().item()
        
        pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float)

        
        _, train_loader, test_loader = get_dataloaders(
            x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args
            
        )

        # stampiamo le forme dei tensori
        #print("x_train_unsupervised.shape:", x_train_unsupervised.shape)
        #print("x_train_few_shot.shape:", x_train_few_shot.shape)
        #print("y_train_few_shot.shape:", y_train_few_shot.shape)
        #print("x_test.shape:", x_test.shape)
        #print("y_test.shape:", y_test.shape)

        # ----------------------------
        # Modello + trainer
        # ----------------------------
        nc = x_train_unsupervised.shape[1]
        model = SimpleDiscriminator(nc=nc, nc_out=params['nc_out'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr_D'])
        #criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

        # ----------------------------
        # Training
        # ----------------------------
        trainer.train(train_loader, epochs=num_epochs, verbose=True)
        #trainer.train(
        #    x_pos=x_class1.to(device),
        #    y_pos=y_class1.to(device).float(),
        #    x_neg=x_class0.to(device),
        #    y_neg=y_class0.to(device).float(),
        #    epochs=num_epochs,
        #    batch_size=batch_size,
        #    n_pos=5  # sempre 5 anomalie per batch
       # )

        # ----------------------------
        # Test / Evaluation
        # ----------------------------
        print("\n--- Test ---")
        test_loss, Auc, pr_auc,gmean_macro = trainer.test(test_loader)
        print(f"Iteration {i} - Test Loss: {test_loss:.6f}, AUC: {Auc:.6f}, PR_AUC: {pr_auc:.6f}, Gmean: {gmean_macro:.6f}")
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():

            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)

                # sigmoid -> probabilities
                probs = torch.sigmoid(logits)

                # threshold 0.5
                preds = (probs >= 0.5).float()

                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print("\n=== Classification Report ===")

        print(
            classification_report(
                all_labels,
                all_preds,
                digits=4
            )
        )

        print("=== Confusion Matrix ===")
        print(confusion_matrix(all_labels, all_preds))

        # metriche aggiuntive
        auc_score = roc_auc_score(all_labels, all_probs)
        pr_auc_score = average_precision_score(all_labels, all_probs)

        print(f"\nROC-AUC: {auc_score:.6f}")
        print(f"PR-AUC : {pr_auc_score:.6f}")





def main():

    args = parse_arguments()

    params = vars(args)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f'Device: {device}')

    params['device'] = device
    params['seed'] = 42
    print("CALLING RUN")

    run(params, args)


if __name__ == '__main__':
    main()


        
