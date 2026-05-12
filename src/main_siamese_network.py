import torch
import time
import pickle
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.siamese_net.trainer import SiameseTrainer
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility, print_args
from siamese_net.dataloader import DynamicPairDataset
from siamese_net.model import SiameseNetwork, Classifier
from siamese_net.trainer import SiameseTrainer, ClassifierTrainer

def main():
    args = parse_arguments()
    print_args(args)

    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    _, train_few_shot_loader, test_loader = get_dataloaders(x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args)

    train_dataset_siamese = DynamicPairDataset(x_train_few_shot, y_train_few_shot, normal_samples_per_epoch=10)
    train_loader_siamese = DataLoader(train_dataset_siamese, batch_size=args.batch_size, shuffle=True)

    set_reproducibility(args.seed)
    attack_type = args.attack_type
    batch_size = args.batch_size


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    input_size = x_train_unsupervised.shape[1]
    y_np = y_train_few_shot.numpy()

    num_pos = (y_np == 1).sum()
    num_neg = (y_np == 0).sum()
    
    pos_weight = torch.tensor(
        num_neg / num_pos,
        dtype=torch.float,
        device=device
    )

    siamese_model = SiameseNetwork(nc = input_size, embedding_dim=args.z_dim).to(device)
    siamese_optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001)
    siamese_criterion = nn.BCELoss()

    siamese_folder = os.path.join(args.SAVE_FOLDER, 'siamese_network')
    os.makedirs(siamese_folder, exist_ok=True)
    siamese_model_path = os.path.join(args.SAVE_FOLDER, 'siamese_network', f'last_siamese_network_{args.attack_type}_{args.n_exps}.pt')
    siamese_loss_path = os.path.join(args.SAVE_FOLDER, 'siamese_network', f'siamese_losses_{args.attack_type}_{args.n_exps}.pdf')

    classifier_path = os.path.join(args.SAVE_FOLDER, 'siamese_network', f'last_classifier_{args.attack_type}_{args.n_exps}.pt')
    classifier_loss_path = os.path.join(args.SAVE_FOLDER, 'siamese_network', f'classifier_losses_{args.attack_type}_{args.n_exps}.pdf')

    siamese_trainer = SiameseTrainer(siamese_model, siamese_optimizer, siamese_criterion, device)

    print(f"Starting {attack_type} Siamese Network model training...")
    start = time.time()
    siamese_trainer.fit(train_dataset_siamese, train_loader_siamese, siamese_model_path, siamese_loss_path,
                        args.n_epochs_siamese_net)
    end = time.time()
    training_time_min_siamese = (end-start)/60
    print("Siamese Network done!")
    print(f"Training time: {end - start:.2f} seconds")

    classifier = Classifier(embedding_dim=args.z_dim)
    classifier_optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.0001)
    
    classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print('Load embedder ...')
    siamese_model.load_state_dict(torch.load(siamese_model_path))
 


    #classifier_optimizer = torch.optim.Adam(params, lr=0.0001)
    classifier_trainer = ClassifierTrainer(classifier, classifier_optimizer, classifier_criterion,
                                           siamese_model.embedder, params=vars(args), device=device)

    print(f"Starting {attack_type} Classifier model training...")
    start = time.time()
    classifier_trainer.fit(train_few_shot_loader, classifier_path, classifier_loss_path, args.n_epochs_classifier_net)
    end = time.time()
    training_time_class = (end-start)/60
    training_time_total = training_time_min_siamese+training_time_class
    print("Classifier done!")
    print(f"Training time: {end - start:.2f} seconds")


    print('Evaluate with Last Model')

    classifier.load_state_dict(torch.load(classifier_path))
    siamese_model.load_state_dict(torch.load(siamese_model_path))
    print(f"Starting Classifier testing on train set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = classifier_trainer.evaluate(classifier,
                                                                                                       siamese_model.embedder,
                                                                                                       train_few_shot_loader,
                                                                                                       classifier_criterion,
                                                                                                       evaluation_on="train")
    print("Classifier test results:")
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc}, pr_auc: {pr_auc}, gmean_macro: {gmean_macro}, Confusion Mat: {cm}, FAR: {fpr}")
    print(cr)

    print("-" * 100)

    print(f"Starting Classifier testing on test set...")
    accuracy, precision, recall, f1, auc, cr, pr_auc, gmean_macro,cm,fpr = classifier_trainer.evaluate(classifier,
                                                                                                       siamese_model.embedder,
                                                                                                       test_loader,
                                                                                                       classifier_criterion,
                                                                                                       evaluation_on="test")
    print("Classifier test results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc}\npr_auc: {pr_auc} \n gmean_macro: {gmean_macro} \n Confusion Mat: {cm} \n FAR: {fpr}")
    print(cr)

    row = {
    "run_id": args.n_runs,   
    "attack_type": args.attack_type,
    "model": "siamese_classifier",

    "accuracy_last": accuracy,
    "precision_last": precision,
    "recall_last": recall,
    "f1_last": f1,
    "auc_last": auc,
    "pr_auc_last": pr_auc,
    "gmean_macro_last": gmean_macro,
    "fpr_last": fpr,

    "training_time_siamese": round(training_time_min_siamese, 3),
    "training_time_class": round(training_time_class, 3),
    "training_time_total": round(training_time_total, 3)
}

    results_dir = os.path.join(args.SAVE_FOLDER, f"run_siamese_{args.n_exps}", args.attack_type)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame([row])

    save_path = os.path.join(results_dir, f"run_last_model_{args.n_runs}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved results to: {save_path}")



if __name__ == '__main__':
    main()