import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    average_precision_score, accuracy_score, precision_recall_fscore_support, precision_recall_curve,auc
)

class SupervisedTrainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'train_loss': []}

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        return total_loss / len(dataloader)

    def fit(self, train_loader, epochs=50):
        print(f"Inizio Training Supervisionato (MLP)...")
        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader, epoch)
            self.history['train_loss'].append(loss)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

    def test(self, test_loader):
            self.model.eval()
            all_probs = []
            all_labels = []
            
            print("\nFase di Test...")
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc="Testing"):
                    x = x.to(self.device)
                    outputs = self.model(x)
                    probs = torch.sigmoid(outputs)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            
            probs = np.array(all_probs)
            y_true = np.array(all_labels)
            y_pred = (probs > 0.5).astype(int)
    
            # 1. AUC-ROC
            roc_auc = roc_auc_score(y_true, probs)
    
            # 2. PR-AUC (Utilizzando precision_recall_curve e auc)
            precision_points, recall_points, _ = precision_recall_curve(y_true, probs)
            pr_auc = auc(recall_points, precision_points)
    
            # 3. Altre Metriche (Soglia 0.5)
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0 
            g_mean = np.sqrt(tpr * tnr)
    
            print("\n" + "="*60)
            print(f" RISULTATI TEST SET (MLP CLASSIFIER)")
            print("="*60)
            print(f"AUC-ROC:  {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
            print(f"Accuracy: {acc:.4f} ")
            print(f"G-Mean:    {g_mean:.4f} | FAR (FPR): {far:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4))
            print("="*60)
    
            return probs, y_true