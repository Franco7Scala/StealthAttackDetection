import torch
import numpy as np
import torch.nn as nn
import os
from typing import Optional
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, auc,roc_curve,confusion_matrix
from torch.utils.data import DataLoader

from src.support.utils import get_base_dir


class model(nn.Module):

    def __init__(self, model3, input_size, output_size, device,params,random_noise=True, mean=0., std=1.):
        super(model, self).__init__()
        self.params = params
        self.random_noise = random_noise
        self.mean = mean
        self.std = std
      
        self.device = device
        self.model3 = model3
        self.fully_connected_1 = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )
        self.to(self.device)
        self.attack_type = self.params['attack_type']      # prende direttamente il valore da params
        self.save_folder = self.params['SAVE_FOLDER']
        self.n_exps = self.params['n_exps']
        self.loss_dir = os.path.join(self.save_folder, 'loss_model', f'loss_model_{self.attack_type}_{self.n_exps}')
        self.prc_dir  = os.path.join(self.save_folder, 'prc_model',  f'prc_model_{self.attack_type}_{self.n_exps}')
        self.auc_dir  = os.path.join(self.save_folder, 'auc_model',  f'auc_model_{self.attack_type}_{self.n_exps}')
        self.probs_csv_dir = os.path.join(self.save_folder, 'output_probs_model_csv', f'output_probs_{self.attack_type}_{self.n_exps}')
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.auc_dir, exist_ok=True)
        os.makedirs(self.prc_dir, exist_ok=True)
        os.makedirs(self.probs_csv_dir, exist_ok=True)

    def forward(self, x):
        if self.random_noise:
            x = x + torch.randn(x.size()).to(x.device) * self.std + self.mean

        _, x3, _ = self.model3.encode(x) #VAE network
        
        x = x3
        logits = self.fully_connected_1(x)
        return logits.flatten()

# -----train and test-----#
    
    def _train_one_epoch_balanced(self, train_loader, optimizer, criterion, batch_size, min_budget = 5):
        self.train()

        self.model3.eval()
        


        for param in self.model3.parameters():
            param.requires_grad = False
    

        x_all = []
        y_all = []
        for x, y in train_loader:
            x_all.append(x)
            y_all.append(y)
        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
    
        # separa anomalie e normali
        mask_pos = (y_all == 1)
        mask_neg = (y_all == 0)
        x_pos = x_all[mask_pos].to(self.device)
        y_pos = y_all[mask_pos].to(self.device)
        x_neg = x_all[mask_neg].to(self.device)
        y_neg = y_all[mask_neg].to(self.device)
        n_pos = x_pos.size(0)
        current_batch_size = int((n_pos / min_budget) * batch_size)
    
        n_neg = current_batch_size - n_pos
        n_batches = len(x_neg) // n_neg

        epoch_loss = 0.0

        for i in tqdm(range(n_batches)):
     
            xb_pos = x_pos
            yb_pos = y_pos
    

            #idx_neg = torch.randperm(len(x_neg))[:n_neg]
            idx_neg = torch.randint(high=len(x_neg), size=(n_neg,), device=self.device)
            xb_neg = x_neg[idx_neg]
            yb_neg = y_neg[idx_neg]
    
            # batch completo
            x_batch = torch.cat([xb_pos, xb_neg], dim=0)
            y_batch = torch.cat([yb_pos, yb_neg], dim=0).float()
    
            # shuffle batch
            perm = torch.randperm(len(y_batch))
            x_batch = x_batch[perm]
            y_batch = y_batch[perm]
    
            # forward/backward
            optimizer.zero_grad()
            logits = self(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        epoch_loss /= n_batches
        print(f"Epoch loss: {epoch_loss:.6f}")
        #_, _, _, _, auc_, _, _, _,_,_ = self.evaluate(train_loader, criterion)
        #print(f"Auc: {auc_}")
        return epoch_loss


    def evaluate(self, loader, criterion, evaluation_on="test"):
        accuracy_am = AverageMeter('Accuracy', ':6.2f')
        precision_am = AverageMeter('Precision', ':6.2f')
        recall_am = AverageMeter('Recall', ':6.2f')
        f1_am = AverageMeter('F1', ':6.2f')
        all_preds = []
        all_targets = []
        all_pred_probs = []
        self.eval()
        self.no_grad = True
        output_probs = []
        for i, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device).view(-1)

            with torch.no_grad():
                logits = self(x)
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
        cm = confusion_matrix(all_targets,all_preds)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn)

        rc_precision, rc_recall, rc_thresholds = precision_recall_curve(all_targets, all_pred_probs)
        fpr, tpr, _ = roc_curve(all_targets, all_pred_probs)
        pr_auc = auc(rc_recall, rc_precision)

        recalls_per_class = recall_score(all_targets, all_preds, average=None)
        gmean_macro = np.prod(recalls_per_class) ** (1 / len(recalls_per_class))

        #---------------------------------------
        # Step 7: Plot the Precision-Recall curve.
        plt.plot(rc_recall, rc_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig(os.path.join(self.prc_dir, f'pr_curve_{evaluation_on}.png'))

        #Step 8: Plot the ROC-AUC-CURVE
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.auc_dir, f'roc_curve_{evaluation_on}.png'))
        plt.close()
        
        #plt.show()
        # ---------------------------------------

        precision_am.update(precision)
        recall_am.update(recall)
        f1_am.update(f1)

        self.no_grad = False


        #string_csv = "\n".join([str(x) for x in output_probs])

        #with open(f"{get_base_dir()}/output_probs_{evaluation_on}.csv", "w") as f:
          #  f.write(string_csv)

        
        # Creiamo un array con le colonne: y_true, y_pred_prob
        output_array = np.column_stack((all_targets, all_pred_probs))
        
        # Salviamo il CSV
        csv_path = os.path.join(self.probs_csv_dir, f'output_{evaluation_on}.csv')
        np.savetxt(csv_path, output_array, delimiter=',', header='y_true,y_pred_prob', comments='')
        
        print(f"[INFO] Saved predictions and labels to {csv_path}")


        return accuracy_am.avg, precision_am.avg, recall_am.avg, f1_am.avg, auc_, cr, pr_auc, gmean_macro,cm,far

    def fit(self, epochs, optimizer, criterion, train_loader,batch_size, best_model_path="", last_model_path="", test_loader: Optional[DataLoader] = None):
        train_losses_per_epoch = []
        best_auprc = -np.inf
        accuracy, precision, recall, f1 = 0, 0, 0, 0
        for epoch in tqdm(range(epochs)):
            avg_loss = self._train_one_epoch_balanced(train_loader, optimizer, criterion,batch_size,min_budget=5)
            train_losses_per_epoch.append(avg_loss)
            
            _, _, _, _, auc_, _, pr_auc, _,_,_ = self.evaluate(train_loader, criterion)
            print(f"Auc: {auc_}, AUPRC: {pr_auc}")

            if pr_auc > best_auprc:
                print('Save best model')
                best_auprc = pr_auc
                torch.save(self.state_dict(), best_model_path)

            if test_loader is not None:
                accuracy, precision, recall, f1, auc_, cr, pr_auc, gmean_macro,cm,far = self.evaluate(test_loader, criterion)

        print("Finished training CPVAE!")
        if test_loader is not None:
            print("Final results:")
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, auc: {auc_}, pr_auc: {pr_auc}, gmean_macro: {gmean_macro}, confusion_mat: {cm}, FAR: {far}")
        self.plotLoss(train_losses_per_epoch)
        torch.save(self.state_dict(), last_model_path)

    def plotLoss(self, loss):
        plt.figure(figsize=(10, 6))
        plt.plot(loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.loss_dir, f'loss_model_training.png'))
        plt.show()
        plt.close()
        

# -----train and test-----#

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