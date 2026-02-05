import torch
import time
import os
import torch.nn as nn
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.support.arguments import parse_arguments

from src.support.utils import set_reproducibility, print_args

from src.Ablation_1.Discriminator import Discriminator
from src.Ablation_1.VAE import VAE, GeneratorLoss
from src.Ablation_1.trainer import Trainer
from src.Ablation_1.model_ablation import ConcatenatedPredictiveVAE



def main():
    args = parse_arguments()
    print_args(args)
    set_reproducibility(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack_type = args.attack_type
    batch_size = args.batch_size
    normalization = args.apply_normalization

    # 1. LOAD DATA
    # x_train_unsupervised serve per il VAE
    # x_train_few_shot serve per il modello ibrido
    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    
    # Dataloader per il pre-training (solo dati normali/non supervisionati)
    train_unsupervised_loader, train_few_shot_loader, test_loader = get_dataloaders(
        x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args
    )

    input_size = x_train_unsupervised.shape[1]

    # --- FASE 1: PRE-TRAINING VAE (UNSUPERVISED) ---
    print(f"\n>>> Fase 1: Pre-training VAE su {device}...")
    if normalization:
        vae_model = VAE(nf_in=input_size, nf_out=args.nf_out, z_dim=args.z_dim,out_activation=nn.ReLU()).to(device)
    else:
        vae_model = VAE(nf_in=input_size, nf_out=args.nf_out, z_dim=args.z_dim).to(device)
        
    
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr_G)
    vae_loss_fn = GeneratorLoss(device=device)
    
    # Usiamo il Trainer creato in precedenza
    vae_trainer = Trainer(vae_model, vae_optimizer, vae_loss_fn, device, save_path=os.path.join(args.SAVE_FOLDER, 'models'))
    

    vae_trainer.fit(train_unsupervised_loader, epochs=args.num_epochs) 
    
    # Carichiamo l'ultimo modello salvato dal trainer per sicurezza
    path_VAE_model = os.path.join(args.SAVE_FOLDER, 'models', 'last_model_vae_abl.pth')
    checkpoint_vae = torch.load(path_VAE_model)
    vae_model.load_state_dict(checkpoint_vae['model_state_dict'])
    print("Pre-training VAE completato e pesi caricati.")


    # --- FASE 2: TRAINING MODELLO IBRIDO (SUPERVISED) ---
    print(f"\n>>> Fase 2: Training Modello Ibrido (CPVAE) - Attacco: {attack_type}...")
    
    # Istanziamo il Discriminatore da zero (model1)
    MC_model = Discriminator(nc=input_size, nc_out=args.nc_out, nout=args.nout).to(device)
    
    # Creiamo il modello ibrido (CPVAE)
    
    CPVAE_model = ConcatenatedPredictiveVAE(
        model1=MC_model, 
        model3=vae_model, 
        input_size=(args.z_dim + args.nc_out), 
        output_size=1, 
        device=device,
        params=vars(args),
        random_noise=True, 
        std=0.05
    )

    # Ottimizzatore solo per le parti non congelate (Disc + Head)
    CPVAE_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CPVAE_model.parameters()), lr=0.0001)
    CPVAE_criterion = nn.BCEWithLogitsLoss()
    our_model_folder = os.path.join(args.SAVE_FOLDER, 'our_models')
    os.makedirs(our_model_folder, exist_ok=True)
    last_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'last_our_models_{args.attack_type}_{args.n_exps}.pt')
    best_model_path = os.path.join(args.SAVE_FOLDER, 'our_models', f'best_our_models_{args.attack_type}_{args.n_exps}.pt')

    start = time.time()
    # Il metodo fit del CPVAE gestisce il bilanciamento e il congelamento internamente
    CPVAE_model.fit(args.n_epochs_cpvae, CPVAE_optimizer, CPVAE_criterion, train_few_shot_loader, batch_size,best_model_path=best_model_path, last_model_path=last_model_path)
    end = time.time()

    print(f"\nTraining Ibrido completato in: {end - start:.2f} secondi")

    #EVALUATE LAST MODEL
    print('Evaluate with Last Model')

    CPVAE_model.load_state_dict(torch.load(last_model_path))
    print(f"Starting ConcatenatedPredictiveVAE testing on train set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = CPVAE_model.evaluate(train_few_shot_loader, CPVAE_criterion,
                                                                            evaluation_on="train")
    

    # --- VALUTAZIONE FINALE ---
    print("\n>>> Valutazione Finale su Test Set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro, cm, fpr = CPVAE_model.evaluate(
        test_loader, CPVAE_criterion, evaluation_on="test"
    )
    
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"FAR:       {fpr:.4f}")
    print(f"pr_auc:       {pr_auc:.4f}")
    print(f"gmean:       {gmean_macro:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(cr)

    print('Evaluate with Best Model')

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