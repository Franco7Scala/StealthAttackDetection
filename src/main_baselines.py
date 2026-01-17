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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.support import utils
from src.support.utils import set_reproducibility, compute_scale_pos_weight
from src.dataset.dataset_loader import load_dataset
from src.support.arguments import parse_arguments


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
    precision = precision_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")
    auc_score = roc_auc_score(y_test, pred, average="weighted")
    pred_prob = model.predict_proba(x_test)
    recalls_per_class = recall_score(y_test, pred, average=None)
    gmean_macro = np.prod(recalls_per_class) ** (1 / len(recalls_per_class))
    #rc_precision, rc_recall, rc_thresholds = precision_recall_curve(y_test, pred_prob[:, 1])
    #pr_auc = auc(rc_recall, rc_precision)
    print("Results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc_score}\ngmean_macro: {gmean_macro}")#\npr_auc: {pr_auc}")
    print(classification_report(y_test, pred, target_names=["Benign", "Attack"]))


if __name__ == "__main__":

    args = parse_arguments()
    set_reproducibility(args.seed)

    # Carica il dataset per calcolare lo sbilanciamento
    _, x_train_few_shot, y_train_few_shot, _, _ = load_dataset(args)
    y_train_np = y_train_few_shot.numpy()

    # Calcolo scale_pos_weight
    scale_pos_weight = compute_scale_pos_weight(y_train_np)

    
    models = [GaussianNB(), DecisionTreeClassifier(max_depth=3, class_weight='balanced'), KNeighborsClassifier(n_neighbors=3),
              RandomForestClassifier(n_estimators=80, class_weight='balanced'), xgb.XGBClassifier(base_score=0.5, n_estimators=80,scale_pos_weight=scale_pos_weight)]
    
    args = parse_arguments()
    for model in models:
        set_reproducibility(args.seed)
        train_model(model, args)
        print("-" * 100)

    print("All models trained and evaluated!")
