import numpy as np
import torch
import os

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

from src.arn.plotter import plot_pr_curve, plot_auc_curve


def generate_labels(size, pflip, lb, ub, step, decay=.9995, up=True): # .9994536323918296

    if up:
        lb = ub - (ub-lb)*((decay)**step)
    else:
        ub = lb + (ub-lb)*((decay)**step)
    pflip = pflip*((decay)**step)

    y = np.random.uniform(lb, ub,size)

    sf = int(pflip*size)
    if sf > 0:
        y[:sf] = 1- y[:sf]
        np.random.shuffle(y)

    return torch.FloatTensor(y)


def predict(D, device, test_loader):
    D.eval()
    i = 0

    for batch, label in test_loader:
        batch = batch.to(device)
        label = label.to(device)

        with torch.no_grad():
            y_pred = D(batch)

        if i == 0:
            y_true = label.cpu()
            yP = y_pred.cpu()
        else:
            y_true = torch.cat((y_true, label.cpu()))
            yP = torch.cat((yP, y_pred.cpu()))

        i += 1

    return y_true, yP


def save_arn_models(G, D, path_G, path_D):
    torch.save(D.state_dict(), path_D)
    torch.save(G.state_dict(), path_G)


def load_arn_models(G, D, path_G, path_D):
    if os.path.exists(path_G):
        G.load_state_dict(torch.load(path_G))

    if os.path.exists(path_D):
        D.load_state_dict(torch.load(path_D))

def get_auprc(y_test, y_pred, show_pr_curve = True):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc_score = auc(recall, precision)
    print(f'AUPRC: {auprc_score:.2f}')

    if show_pr_curve:
        plot_pr_curve(precision, recall)
    return auprc_score

def get_auc(y_test, y_pred, show_auc_curve = True):
    auc_score = roc_auc_score(y_test, y_pred)
    print(f'AUC: {auc_score:.2f}')

    if show_auc_curve:
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plot_auc_curve(fpr, tpr)

    return auc_score

