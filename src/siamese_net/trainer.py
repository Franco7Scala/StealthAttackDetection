import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc,roc_curve,confusion_matrix


class SiameseTrainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False)

        for x1, x2, y in pbar:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device).float()

            self.optimizer.zero_grad()
            z1, z2 = self.model(x1, x2)

            distance = F.pairwise_distance(z1, z2)
            scores = torch.exp(-distance)

            loss = self.criterion(scores, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / len(dataloader)

    def fit(self, train_dataset, train_loader, last_model_path, loss_path, epochs=50):
        losses = []
        for epoch in range(1, epochs + 1):
            train_dataset.resample_pairs()
            loss = self.train_one_epoch(train_loader, epoch)
            losses.append(loss)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

        torch.save(self.model.state_dict(), last_model_path)
        self.plotLoss(losses, loss_path)

    def plotLoss(self, loss, loss_path):
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_path)
        plt.show()
        plt.close()


class ClassifierTrainer:
    def __init__(self, model, optimizer, criterion, embedder, params, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.embedder = embedder
        self.device = device
        self.history = {'train_loss': []}
        self.params = params

        self.attack_type = self.params['attack_type']
        self.save_folder = self.params['SAVE_FOLDER']
        self.n_exps = self.params['n_exps']

        self.prc_dir = os.path.join(self.save_folder, 'prc_model', f'prc_model_{self.attack_type}_{self.n_exps}')
        self.auc_dir = os.path.join(self.save_folder, 'auc_model', f'auc_model_{self.attack_type}_{self.n_exps}')
        self.probs_csv_dir = os.path.join(self.save_folder, 'output_probs_model_csv',
                                          f'output_probs_{self.attack_type}_{self.n_exps}')

        os.makedirs(self.auc_dir, exist_ok=True)
        os.makedirs(self.prc_dir, exist_ok=True)
        os.makedirs(self.probs_csv_dir, exist_ok=True)

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        self.embedder.eval()

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False)

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            z = self.embedder(x)
            logits = self.model(z)

            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / len(dataloader)

    def fit(self, train_loader, last_model_path, loss_path, epochs=50):
        losses = []
        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader, epoch)
            losses.append(loss)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

        torch.save(self.model.state_dict(), last_model_path)
        self.plotLoss(losses, loss_path)

    def plotLoss(self, loss, loss_path):
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_path)
        plt.show()
        plt.close()

    def evaluate(self, model, embedder, loader, criterion, evaluation_on="test"):
        accuracy_am = AverageMeter('Accuracy', ':6.2f')
        precision_am = AverageMeter('Precision', ':6.2f')
        recall_am = AverageMeter('Recall', ':6.2f')
        f1_am = AverageMeter('F1', ':6.2f')

        model.eval()
        embedder.eval()

        all_preds = []
        all_targets = []
        all_pred_probs = []


        output_probs = []

        for i, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device).view(-1)

            with torch.no_grad():
                z = embedder(x)
                logits = model(z)
                loss = criterion(logits, y)

            y_pred_prob = torch.sigmoid(logits)
            y_pred = (y_pred_prob > 0.5) + 0.
            accuracy = accuracy_score(y.cpu(), y_pred.cpu())
            accuracy_am.update(accuracy, x.size(0))
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_pred_probs.extend(y_pred_prob.cpu().numpy())

            tp_output_probs = y_pred_prob.detach().cpu().numpy()
            tp_ground_truth = y.cpu().numpy()
            tp_concatenated = np.concatenate((tp_output_probs, tp_ground_truth))
            output_probs.extend(tp_concatenated.tolist())

        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        auc_ = roc_auc_score(y_true=all_targets, y_score=all_pred_probs)
        cr = classification_report(all_targets, all_preds, target_names=["Benign", "Attack"])
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn)

        rc_precision, rc_recall, rc_thresholds = precision_recall_curve(all_targets, all_pred_probs)
        fpr, tpr, _ = roc_curve(all_targets, all_pred_probs)
        pr_auc = auc(rc_recall, rc_precision)

        recalls_per_class = recall_score(all_targets, all_preds, average=None)
        gmean_macro = np.prod(recalls_per_class) ** (1 / len(recalls_per_class))

        # ---------------------------------------
        # Step 7: Plot the Precision-Recall curve.
        plt.plot(rc_recall, rc_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig(os.path.join(self.prc_dir, f'pr_curve_{evaluation_on}.png'))

        # Step 8: Plot the ROC-AUC-CURVE
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.auc_dir, f'roc_curve_{evaluation_on}.png'))
        plt.close()

        # plt.show()
        # ---------------------------------------

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        # Creiamo un array con le colonne: y_true, y_pred_prob
        output_array = np.column_stack((all_targets, all_pred_probs))

        # Salviamo il CSV
        csv_path = os.path.join(self.probs_csv_dir, f'output_{evaluation_on}.csv')
        np.savetxt(csv_path, output_array, delimiter=',', header='y_true,y_pred_prob', comments='')

        print(f"[INFO] Saved predictions and labels to {csv_path}")

        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg, auc_, cr, pr_auc, gmean_macro, cm, far


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
