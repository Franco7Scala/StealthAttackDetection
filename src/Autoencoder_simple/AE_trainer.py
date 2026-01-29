import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, 
    auc, confusion_matrix, average_precision_score,accuracy_score, precision_recall_fscore_support
)

class AE_Trainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.thresholds_dict = {}
        self.history = {'train_loss': []}

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False)
        
        for x in pbar:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            reconstruction = self.model(x)
            loss = self.criterion(reconstruction, x)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        return total_loss / len(dataloader)

    def fit(self, train_loader, epochs):
        print(f"Inizio Training su {self.device}...")
        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader, epoch)
            self.history['train_loss'].append(loss)
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")
        
        self.compute_all_thresholds(train_loader)

    def compute_all_thresholds(self, dataloader):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                reconstruction = self.model(x)
                loss = torch.mean((reconstruction - x)**2, dim=1)
                errors.extend(loss.cpu().numpy())
        
        errors = np.array(errors)
        mu = np.mean(errors)
        sigma = np.std(errors)
        
        # Configurazione alpha come richiesto
        alphas = [2.75, 3.0, 3.25]
        self.thresholds_dict = {f'alpha_{a}': mu + (a * sigma) for a in alphas}
        for k, v in self.thresholds_dict.items(): print(f" - {k}: {v:.6f}")

        
    def test(self, test_loader):
        self.model.eval()
        errors = []
        all_labels = []
        
        print("\nFase di Test: calcolo errore di ricostruzione...")
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.to(self.device)
                reconstruction = self.model(x)
                loss = torch.mean((reconstruction - x)**2, dim=1)
                errors.extend(loss.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        errors = np.array(errors)
        y_true = np.array(all_labels)

        # Metriche Globali
        roc_auc = roc_auc_score(y_true, errors)
        pr_auc = average_precision_score(y_true, errors)

        print("\n" + "="*60)
        print(f" METRICHE GLOBALI (Indipendenti dalla soglia)")
        print("="*60)
        print(f"AUC-ROC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        # Liste per calcolare le medie
        metrics_log = {
            'acc': [], 'f1': [], 'prec': [], 'rec': [], 'gmean': [], 'far': []
        }

   

        for name, thresh in self.thresholds_dict.items():
            y_pred = (errors > thresh).astype(int)
            
            # Calcolo metriche puntuali
            acc = accuracy_score(y_true, y_pred)
            # Usiamo macro averaging per le metriche di classificazione
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0 
            g_mean = np.sqrt(tpr * tnr)

            # Log dei risultati
            metrics_log['acc'].append(acc)
            metrics_log['f1'].append(f1)
            metrics_log['prec'].append(prec)
            metrics_log['rec'].append(rec)
            metrics_log['gmean'].append(g_mean)
            metrics_log['far'].append(far)

            print(f"\n>>> SOGLIA {name.upper()} (val: {thresh:.6f})")
            print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4))
            print(f"G-Mean:   {g_mean:.4f} | FAR:       {far:.4f} | Accuracy: {acc:.4f}")

        # --- STAMPA DELLE MEDIE FINALI ---
        print("\n" + "="*60)
        print(f" RIEPILOGO MEDIATO SU TUTTE LE SOGLIE ({', '.join(self.thresholds_dict.keys())})")
        print("="*60)
        print(f"Media Accuracy:  {np.mean(metrics_log['acc']):.4f}")
        print(f"Media F1-Score:  {np.mean(metrics_log['f1']):.4f}")
        print(f"Media Precision: {np.mean(metrics_log['prec']):.4f}")
        print(f"Media Recall:    {np.mean(metrics_log['rec']):.4f}")
        print(f"Media G-Mean:    {np.mean(metrics_log['gmean']):.4f}")
        print(f"Media FAR:       {np.mean(metrics_log['far']):.4f}")
        print("="*60)

        return errors, y_true
        
    def test_few_shot(self, test_loader, x_attack_few_shot):
        """
        calcola la soglia basata sul minimo errore
        di ricostruzione dei campioni di attacco forniti.
        """
        self.model.eval()
        
        # --- 1. Calcolo della Soglia Custom ---
        print("\nCalcolo soglia basata sul minimo errore dei campioni di attacco...")
        with torch.no_grad():
            if not isinstance(x_attack_few_shot, torch.Tensor):
                x_attack_few_shot = torch.tensor(x_attack_few_shot, dtype=torch.float32)
            
            x_attack_few_shot = x_attack_few_shot.to(self.device)
            recon_atk = self.model(x_attack_few_shot)
            # Calcoliamo l'errore per ogni sample di attacco
            atk_errors = torch.mean((recon_atk - x_attack_few_shot)**2, dim=1)
            
            # La soglia è il valore minimo di errore tra gli attacchi conosciuti
            custom_thresh = torch.min(atk_errors).item()
            print(f"Soglia Min-Attack calcolata: {custom_thresh:.6f}")

        # --- 2. Fase di Test ---
        errors = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Few-Shot Testing"):
                x = x.to(self.device)
                reconstruction = self.model(x)
                loss = torch.mean((reconstruction - x)**2, dim=1)
                errors.extend(loss.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        errors = np.array(errors)
        y_true = np.array(all_labels)
        
        # Classificazione binaria con la nuova soglia
        y_pred = (errors > custom_thresh).astype(int)

        # --- 3. Calcolo Metriche ---
        roc_auc = roc_auc_score(y_true, errors)
        # PR-AUC usando precision_recall_curve e auc
        prec_pts, rec_pts, _ = precision_recall_curve(y_true, errors)
        pr_auc = auc(rec_pts, prec_pts)

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_mean = np.sqrt(tpr * tnr)

        print("\n" + "="*60)
        print(f" RISULTATI FEW-SHOT TEST (Soglia: {custom_thresh:.6f})")
        print("="*60)
        print(f"AUC-ROC:  {roc_auc:.4f} | PR-AUC:   {pr_auc:.4f}")
        print(f"F1-Score: {f1:.4f} | G-Mean:   {g_mean:.4f}")
        print(f"FAR:      {far:.4f} | Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4))
        print("="*60)

        return errors, y_true