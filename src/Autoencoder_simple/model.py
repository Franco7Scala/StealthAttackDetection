import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):

    def __init__(self, nc=16, n_latent=8, nout=16):
        super(SimpleAutoencoder, self).__init__()
        self.nc = nc          # Input/Output dimension
        self.n_latent = n_latent  # Latent space dimension
        self.nout = nout      # Hidden layers dimension

        # --- ENCODER ---
        
        self.encoder = nn.Sequential(
            nn.Linear(self.nc, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.ReLU(),

            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.Dropout(0.05),
            nn.ReLU(),

            nn.Linear(self.nout, self.n_latent),
            nn.BatchNorm1d(self.n_latent),
            nn.ReLU(),
        )

        # --- DECODER ---

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.ReLU(),

            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.Dropout(0.05),
            nn.ReLU(),

            nn.Linear(self.nout, self.nc),

        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        latent = self.encoder(x)
 
        reconstruction = self.decoder(latent)
        return reconstruction


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)