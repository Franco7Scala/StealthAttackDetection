import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import warnings
warnings.filterwarnings("ignore")

import pickle
import time
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.support import utils
from src.support.utils import set_reproducibility, compute_scale_pos_weight
from src.dataset.dataset_loader import load_dataset
from src.support.arguments import parse_arguments
import pandas as pd

def train_model(model, args):
    print("Loading dataset...")

    _, x_train_few_shot, y_train_few_shot, x_test, y_test = load_dataset(args)
    x_train_few_shot = x_train_few_shot.numpy()
    y_train_few_shot = y_train_few_shot.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    model_name = type(model).__name__
    print(f"Training {model_name} model...")
    start = time.time()
    model.fit(x_train_few_shot, y_train_few_shot if model_name == "XGBClassifier" else y_train_few_shot.ravel())
    end = time.time()
    print(f"Training finished in: {end - start:.2f} seconds!")
    print(f"Evaluating model...")
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro")
    recall = recall_score(y_test, pred, average="macro")
    f1 = f1_score(y_test, pred, average="macro")
    
    pred_prob = model.predict_proba(x_test)
    recalls_per_class = recall_score(y_test, pred, average=None)
    gmean_macro = np.prod(recalls_per_class) ** (1 / len(recalls_per_class))
    rc_precision, rc_recall, rc_thresholds = precision_recall_curve(y_test, pred_prob[:, 1])
    pr_auc = auc(rc_recall, rc_precision)
    auc_score = roc_auc_score(y_test, pred_prob[:, 1], average="macro")
    cm = confusion_matrix(y_test,pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    print("Results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc_score}\ngmean_macro: {gmean_macro}\npr_auc: {pr_auc}\nConfusionMat: {cm}\nFAR: {fpr}")
    print(classification_report(y_test, pred, target_names=["Benign", "Attack"]))
   
    return {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "auc": auc_score,
    "gmean_macro": gmean_macro,
    "pr_auc": pr_auc,
    "fpr": fpr,
    "training_time": round(end - start, 3)
}


if __name__ == "__main__":

    args = parse_arguments()
    set_reproducibility(args.seed)
    BASE_DIR = os.path.join(args.SAVE_FOLDER, "run_baselines", args.attack_type)
    os.makedirs(BASE_DIR, exist_ok=True)

    run_id = args.n_runs 

    # Carica il dataset per calcolare lo sbilanciamento
    _, x_train_few_shot, y_train_few_shot, _, _ = load_dataset(args)
    y_train_np = y_train_few_shot.numpy()

    # Calcolo scale_pos_weight
    scale_pos_weight = compute_scale_pos_weight(y_train_np)

    
    #models = [GaussianNB(), DecisionTreeClassifier(class_weight='balanced'), KNeighborsClassifier(n_neighbors=3),
    #          RandomForestClassifier(n_estimators=80, class_weight='balanced'), xgb.XGBClassifier(base_score=0.5, n_estimators=80,scale_pos_weight=scale_pos_weight)]
    
    
    #args = parse_arguments()
    models = [
        ("GaussianNB", GaussianNB()),
        ("DecisionTree", DecisionTreeClassifier(class_weight='balanced')),
        ("KNN", KNeighborsClassifier(n_neighbors=3)),
        ("RandomForest", RandomForestClassifier(n_estimators=80, class_weight='balanced')),
        ("XGBoost", xgb.XGBClassifier(
            base_score=0.5,
            n_estimators=80,
            scale_pos_weight=scale_pos_weight
        ))
    ]
    results = []
    for model_name, model in models:
        print(f"\nTraining {model_name} (run {run_id})")
        set_reproducibility(args.seed)
        metrics = train_model(model, args)
        print("-" * 100)
        results.append({
            "run_id": run_id,
            "attack_type": args.attack_type,
            "model": model_name,
            **metrics
        })

    df = pd.DataFrame(results)

    save_path = os.path.join(BASE_DIR, f"run_{run_id}.csv")
    df.to_csv(save_path, index=False)

    print(f"\nSaved results to: {save_path}")

    print("All models trained and evaluated!")


