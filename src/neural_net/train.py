import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc, classification_report,recall_score, confusion_matrix
import numpy as np
import pandas as pd



class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_losses = []

    def train_one_epoch(self, dataloader):
        
        self.model.train()
        epoch_loss = 0.0
        y_true = []
        y_prob = []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            x = batch[0]
            y = batch[1]
            x = x.to(self.device)
            y = y.to(self.device).float()
            #print("x_batch.shape:", x.shape)
            #print("y_batch.shape:", y.shape)

           
            # forward
            output = self.model(x)
            probs = torch.sigmoid(output).detach().cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs)
            
   
            loss = self.criterion(output, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        df = pd.DataFrame({
                "y_true": y_true,
                "prob_preds": y_prob
            })
        df.to_excel('prediction_train.xlsx', index=False)

        epoch_loss /= len(dataloader)
        return epoch_loss

    def train(self, dataloader, epochs=30, verbose=True):
        for epoch in range(epochs):
            loss = self.train_one_epoch(dataloader)
            self.train_losses.append(loss)

            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.6f}")
                
    def train_balanced(self, x_pos, y_pos, x_neg, y_neg, epochs=30, batch_size=16, n_pos=5):
        self.model.train()
    
        n_neg = batch_size - n_pos
        x_pos = x_pos.to(self.device)
        y_pos = y_pos.to(self.device)
        x_neg = x_neg.to(self.device)
        y_neg = y_neg.to(self.device)
    
        for epoch in range(epochs):
            epoch_loss = 0.0
           
            # shuffle normali
            #perm_neg = torch.randperm(len(x_neg))
            #x_neg = x_neg[perm_neg]
            #y_neg = y_neg[perm_neg]
    
            n_batches = len(x_neg) // n_neg
    
            for i in range(n_batches):
                # anomalie: sempre le stesse 5
                xb_pos = x_pos
                yb_pos = y_pos
            
                # normali: sequenziali
                #start = i * n_neg
                #end = start + n_neg
                #xb_neg = x_neg[start:end]
                #yb_neg = y_neg[start:end]


                #idx_neg = torch.randint(0, len(x_neg), (n_neg,))
                #idx_neg = torch.randperm(len(x_neg))[:n_neg]
                idx_neg = torch.randint(high=len(x_neg), size=(n_neg,), device=self.device)

                xb_neg = x_neg[idx_neg]
                yb_neg = y_neg[idx_neg]
    
                # batch finale
                x_batch = torch.cat([xb_pos, xb_neg], dim=0)
                y_batch = torch.cat([yb_pos, yb_neg], dim=0)
    
                # shuffle del batch
                perm = torch.randperm(len(y_batch))
                x_batch = x_batch[perm]
                y_batch = y_batch[perm]
            
                # batch finale
                #x_batch = torch.cat([xb_pos, xb_neg], dim=0)
                #y_batch = torch.cat([yb_pos, yb_neg], dim=0)
            
                # shuffle del batch (opzionale ma consigliato)
                #perm = torch.randperm(len(y_batch))
                #x_batch = x_batch[perm]
                #y_batch = y_batch[perm]
    
    
                # forward
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
    
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                epoch_loss += loss.item()
    
            epoch_loss /= n_batches
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")


                
    def test(self, dataloader):
        self.model.eval()  
        run_loss = 0.0

        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():  
            for x, y in tqdm(dataloader, desc="Testing", leave=False):
                x = x.to(self.device)
                y = y.to(self.device).float()

                outputs = self.model(x)  
                loss = self.criterion(outputs, y)
                run_loss += loss.item()

                probs = outputs.detach().cpu().numpy()
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds)
                y_prob.extend(probs)

        run_loss /= len(dataloader)

        # metriche
        auc_ = roc_auc_score(y_true, y_prob)
        recalls_per_class = recall_score(y_true, y_pred, average=None)
        gmean_macro = np.prod(recalls_per_class) ** (1 / len(recalls_per_class))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)  # area sotto la curva precision-recall
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        df = pd.DataFrame({
                "y_true": y_true,
                "prob_preds": y_prob
            })
        df.to_excel('prediction_test.xlsx')

        print(" Test Results")
        print(f"Loss media test set: {run_loss:.6f}")
        print(f"AUC:  {auc_:.6f}")
        print(f'PR_AUC: {pr_auc:.6f}')
        print(f"G-Mean (macro): {gmean_macro:.6f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        print("\nConfusion Matrix:")
        print(cm)
        print(f'FAR: {fpr}')
        print(tn,fp,fn,tp)

        return run_loss, auc_, pr_auc, gmean_macro