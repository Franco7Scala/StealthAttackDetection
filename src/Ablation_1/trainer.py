import torch
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, save_path):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_path = save_path
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
      
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}", leave=False)
        
        for x in pbar:
            x = x.to(self.device).float()
            self.optimizer.zero_grad()
            
            logits, mu, logvar, sampled_data = self.model(x)
            loss, recon, kld = self.loss_fn.return_loss(sampled_data, x, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MSE': f"{recon.item():.4f}",
                'KLD': f"{kld.item():.4f}"
            })
            
        num_batches = len(train_loader)
        return total_loss / num_batches, total_recon / num_batches, total_kld / num_batches

    def fit(self, train_loader, epochs):
        # Barra delle epoche (esterna)
        main_pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
        
        for epoch in main_pbar:
            avg_loss, avg_recon, avg_kld = self.train_epoch(train_loader, epoch)
            
            # Aggiorna la barra principale con i risultati dell'ultima epoca
            main_pbar.set_postfix({
                'Last_Loss': f"{avg_loss:.4f}",
                'Last_MSE': f"{avg_recon:.4f}"
            })

            # Salvataggio dell'ultimo modello (sovrascrive)
            checkpoint_path = os.path.join(self.save_path, "last_model_vae_abl.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

        print(f"\nAddestramento completato. Modello finale salvato in: {checkpoint_path}")