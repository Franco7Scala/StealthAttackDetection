from typing import Optional

from tqdm import tqdm
import numpy
import torch
import pandas
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class VAENN(nn.Module):

    def __init__(self, dim_code, input_size, device):
        super().__init__()
        self.label = nn.Embedding(10, dim_code)
        self.device = device
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.flatten_mu = nn.Linear(256, out_features=dim_code)
        self.flatten_log_sigma = nn.Linear(256, out_features=dim_code)
        # decoder
        self.decode_linear = nn.Linear(dim_code, 256)
        self.decode_2 = nn.Linear(256, 128)
        self.decode_1 = nn.Linear(128, input_size)
        self.to(self.device)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.flatten_mu(x), self.flatten_log_sigma(x)
        z = self.gaussian_sampler(mu, log_sigma)
        return z

    def gaussian_sampler(self, mu, log_sigma):
        if self.training:
            std = torch.exp(log_sigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decode(self, x):
        x = self.decode_linear(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return reconstruction

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.flatten_mu(x), self.flatten_log_sigma(x)
        z = self.gaussian_sampler(mu, log_sigma)
        x = self.decode_linear(z)
        x = x.view(x.size(0), 256)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return mu, log_sigma, reconstruction

    def _train_epoch(self, optimizer, data_loader):
        train_losses_per_epoch = []
        recon_losses_per_epoch = []
        self.train()
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(self.device)
            mu, log_sigma, reconstruction = self(x_batch)
            recon_loss = log_likelihood(x_batch.to(self.device).float(), reconstruction)
            kl_loss = kl_divergence(mu, log_sigma)
            loss = kl_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())
            recon_losses_per_epoch.append(recon_loss.item())

        return numpy.mean(train_losses_per_epoch), numpy.mean(recon_losses_per_epoch)

    def evaluate(self, data_loader):
        val_losses_per_epoch = []
        recon_losses_per_epoch = []
        self.eval()
        with torch.no_grad():
            for x_val, _ in data_loader:
                x_val = x_val.to(self.device)
                mu, log_sigma, reconstruction = self(x_val)
                recon_loss = log_likelihood(x_val.to(self.device).float(), reconstruction)
                kl_loss = kl_divergence(mu, log_sigma)
                loss = kl_loss + recon_loss
                val_losses_per_epoch.append(loss.item())
                recon_losses_per_epoch.append(recon_loss.item())

        return numpy.mean(val_losses_per_epoch), numpy.mean(recon_losses_per_epoch)

    def fit(self, epochs, optimizer, train_loader, test_loader: Optional[DataLoader] = None):
        loss = {"train_loss": [], "val_loss": [], "train_recon_loss": [], "val_recon_loss": []}
        #with tqdm(desc="Training", total=epochs) as pbar_outer:
        for epoch in tqdm(range(epochs)):
            train_loss, train_recon_loss = self._train_epoch(optimizer, train_loader)
            if test_loader is not None:
                val_loss, val_recon_loss = self.evaluate(test_loader)
            #pbar_outer.update(1)
            if test_loader is not None:
                loss["val_loss"].append(val_loss)
                loss["val_recon_loss"].append(val_recon_loss)
            loss["train_loss"].append(train_loss)
            loss["train_recon_loss"].append(train_recon_loss)
        self.plotLoss(loss)

    def plotLoss(self, loss):
        plt.figure(figsize=(10, 6))
        plt.plot(loss["train_recon_loss"], label="Train Reconstruction Loss", linewidth=2)
        plt.plot(loss["val_recon_loss"], label="Validation Reconstruction Loss", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Reconstruction Error")
        plt.title("Reconstruction Error trend")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def draw_latent_space(self, test_dataset, path_image):
        latent_space = []
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        for x, y in tqdm(test_loader):
            img = x.to(self.device)
            label = y.to(self.device)
            self.eval()
            with torch.no_grad():
                latent = self.encode(img)

            latent = latent.flatten().cpu().numpy()
            sample = {f"Encoded_{i}": encoded for i, encoded in enumerate(latent)}
            sample["label"] = label.item()
            latent_space.append(sample)

        latent_space = pandas.DataFrame(latent_space)
        latent_space["label"] = latent_space["label"].astype(str)
        tsne = TSNE(n_components=2)
        digits_embedded = tsne.fit_transform(latent_space.drop(["label"], axis=1))
        figure = px.scatter(digits_embedded, x=0, y=1, color=latent_space["label"], opacity=0.7, labels={"color": "Digit"}, title="Latent space with t-SNE").for_each_trace(lambda t: t.update(name=t.name.replace("=", ": ")))
        figure.update_traces(marker=dict(size=10, line=dict(width=2,  color="DarkSlateGrey")), selector=dict(mode="markers"))
        figure.update_yaxes(visible=False, showticklabels=False)
        figure.update_xaxes(visible=False, showticklabels=False)
        figure.show()
        figure.write_image(path_image)


def load_model(dim_code, path, device):
    model = VAENN(dim_code, device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def kl_divergence(mu, log_sigma):
    loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return loss


def log_likelihood(x, reconstruction):
    loss = nn.CrossEntropyLoss(reduction="sum")
    return loss(reconstruction, x)


def loss_vae(x, mu, log_sigma, reconstruction):
    return kl_divergence(mu, log_sigma) + log_likelihood(x, reconstruction)
