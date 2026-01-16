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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.support import utils
from src.support.utils import get_base_dir


def train_model(model):
    print("Loading dataset...")
    x_test = pickle.load(open(f'{get_base_dir()}/pickles/x_test_slowdos.pkl', 'rb'))
    y_test = pickle.load(open(f'{get_base_dir()}/pickles/y_test_slowdos.pkl', 'rb'))
    x_train = pickle.load(open(f'{get_base_dir()}/pickles/x_train_slowdos.pkl', 'rb'))
    y_train = pickle.load(open(f'{get_base_dir()}/pickles/y_train_slowdos.pkl', 'rb'))
    model_name = type(model).__name__
    print(f"Training {model_name} model...")
    start = time.time()
    model.fit(x_train, y_train if model_name == "XGBClassifier" else y_train.ravel())
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
    #rc_precision, rc_recall, rc_thresholds = precision_recall_curve(y_test, pred_prob[:, 1])
    #pr_auc = auc(rc_recall, rc_precision)
    print("Results:")
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\nauc: {auc_score}")#\npr_auc: {pr_auc}")
    print(classification_report(y_test, pred, target_names=["Benign", "SlowDoS"]))


if __name__ == "__main__":
    models = [GaussianNB(), DecisionTreeClassifier(max_depth=3), KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(n_estimators=80), xgb.XGBClassifier(base_score=0.5, n_estimators=80)]
    for model in models:
        utils.seed_everything(1)
        train_model(model)
        print("-" * 100)

    print("All models trained and evaluated!")
