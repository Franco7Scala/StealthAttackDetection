import torch
import time 
import os
import numpy
import torch.nn as nn
import pandas as pd
from src.dataset.dataset_loader import load_dataset, get_dataloaders
from src.model.predictive_model import AverageMeter
from src.prototypical_net.trainer_proto import train_proto
from src.support.arguments import parse_arguments
from src.support.utils import set_reproducibility, print_args
from src.arn.model import Generator
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, recall_score, accuracy_score, roc_auc_score,    precision_score


if __name__ == "__main__":
    args = parse_arguments()
    print_args(args)
    num_normal_prototypes = 1 # k for knn
    use_vae_pretrained = True

    #############################################################################

    x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    _, train_few_shot_loader, test_loader = get_dataloaders(x_train_unsupervised, x_train_few_shot, y_train_few_shot, x_test, y_test, args)
    
    set_reproducibility(args.seed)
    attack_type = args.attack_type
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")
    x_train_unsupervised = x_train_unsupervised.to(device)
    input_size = x_train_unsupervised.shape[1]
    start_time_total = time.time()
    if use_vae_pretrained:
        name_VAE_model = f"ARN_Generator_{attack_type}_0.ckpt"
        path_VAE_model = os.path.join(args.SAVE_FOLDER, "models", name_VAE_model)
        if args.apply_normalization:
          VAE_model = Generator(nf_in=input_size, nf_out=args.nf_out, z_dim=args.z_dim, out_activation=nn.ReLU).to(device)

        else:
            VAE_model = Generator(nf_in=input_size, nf_out=args.nf_out, z_dim=args.z_dim).to(device)

        VAE_model.load_state_dict(torch.load(path_VAE_model))
        VAE_model.eval()

    else:
        VAE_model = train_proto(args, x_train_unsupervised, x_train_few_shot)

    for param in VAE_model.parameters():
        param.requires_grad = False

    def get_embedding(model, x_tensor):
        with torch.no_grad():
            output = model(x_tensor)
            return output[1]

    print("Extracting embedding for normal traffic...")
    z_norm = get_embedding(VAE_model, x_train_unsupervised)
    z_norm_np = z_norm.cpu().numpy()
    #numpy.random.shuffle(z_norm_np)
    #z_norm_np = z_norm_np[:100]
    
   
    print(f"Clustering normal traffic in {num_normal_prototypes} prototypes...")
    kmeans = KMeans(n_clusters=num_normal_prototypes, random_state=args.seed, n_init=10)
    kmeans.fit(z_norm_np)
    training_time_total = time.time() - start_time_total
    norm_prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    print("Calculating prototype of the attack...")
    x_attack_fs_tensor = torch.tensor(x_train_few_shot, dtype=torch.float32).to(device)
    z_attack = get_embedding(VAE_model, x_attack_fs_tensor)
    attack_prototype = torch.mean(z_attack, dim=0, keepdim=True)

    print("Evaluating...")
    accuracy_am = AverageMeter("Accuracy", ":6.2f")
    y_true = []
    y_scores = []
    y_pred = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        z_batch = get_embedding(VAE_model, x_batch)
        # Calculating Euclidean distance from the attack
        dist_to_attack = torch.cdist(z_batch, attack_prototype, p=2).squeeze(1)
        # Calculating Euclidean distance from the normal prototypes
        dist_to_norms = torch.cdist(z_batch, norm_prototypes, p=2)
        # Finding the nearest normal prototype
        min_dist_to_norm, _ = torch.min(dist_to_norms, dim=1)
        # Making prediction
        predictions = (dist_to_attack < min_dist_to_norm).int()

        def stable_attack_score(dist_attack, dist_norm):
            val_attack = -dist_attack
            val_norm = -dist_norm
            max_val = torch.max(val_attack, val_norm)
            exp_attack = torch.exp(val_attack - max_val)
            exp_norm = torch.exp(val_norm - max_val)
            return exp_attack / (exp_attack + exp_norm)

        score_attack = stable_attack_score(dist_to_attack, min_dist_to_norm)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_scores.extend(score_attack.cpu().numpy())
        accuracy = accuracy_score(y_batch.cpu(), predictions.cpu())
        accuracy_am.update(accuracy, x_batch.size(0))

    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    y_scores = numpy.array(y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    acc = accuracy_am.avg
    recalls_per_class = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    g_mean = numpy.prod(recalls_per_class) ** (1 / len(recalls_per_class))
    auc = roc_auc_score(y_true, y_scores)

    row = {
    "run_id": args.n_runs,   
    "attack_type": args.attack_type,
    "model": "prototypical",
    "accuracy_last": acc,
    "precision_last": precision,
    "recall_last": recall,
    "f1_last": f1,
    "auc_last": auc,
    "pr_auc_last": pr_auc,
    "gmean_macro_last": g_mean,
    "fpr_last": far,
    "training_time_total": round(training_time_total, 3)
}

    results_dir = os.path.join(args.SAVE_FOLDER, f"run_prototypical_{args.n_exps}", args.attack_type)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame([row])
    save_path = os.path.join(results_dir, f"run_last_model_{args.n_runs}.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved results to: {save_path}")

    print("\n" + "=" * 45)
    print("Results:")
    print("=" * 45)
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"G-Mean  : {g_mean:.4f}")
    print(f"FAR     : {far * 100:.2f} %")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print("=" * 45)
