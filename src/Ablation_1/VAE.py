import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, nf_in=121, nf_out=32, z_dim=16, out_activation=None):
        super(VAE, self).__init__()

        self.nf_in = nf_in
        self.nf_out = nf_out
        self.z_dim = z_dim
        self.out_activation = out_activation

        self.encoder = nn.Sequential(
            nn.Linear(self.nf_in, self.nf_out),
            nn.BatchNorm1d(self.nf_out),
            nn.ReLU(0.2),

            # nn.Linear(self.nf_out * 2, self.nf_out),
            # nn.BatchNorm1d(self.nf_out, track_running_stats = False),
            # nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            # nn.Linear(self.nf_out, self.nf_out * 2),
            # nn.BatchNorm1d(self.nf_out * 2, track_running_stats = False),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.nf_out, self.nf_in)
        )

        self.fc1 = nn.Linear(self.nf_out, self.nf_out)
        self.fc21 = nn.Linear(self.nf_out, self.z_dim)
        self.fc22 = nn.Linear(self.nf_out, self.z_dim)

        self.fc3 = nn.Linear(self.z_dim, self.nf_out)
        self.fc4 = nn.Linear(self.nf_out, self.nf_out)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        conv = self.encoder(x)
        h = self.fc1(conv)

        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        h = self.relu(self.fc3(z))
        deconv_input = self.fc4(h)

        return self.decoder(deconv_input)

        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        logits = self.decode(z)
    
        if self.out_activation is not None:
            sampled_data = self.out_activation(logits)
        else:
            sampled_data = logits   
    
        return logits, mu, logvar, sampled_data


class GeneratorLoss(nn.Module):
    def __init__(self, device):
        super(GeneratorLoss, self).__init__()
        self.device = device
      
        self.mse = nn.MSELoss(reduction='mean')

    def KLD(self, mu, logvar):

        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        #kld = torch.mean(0.5 * (-0.5 * logvar + torch.exp(0.5 * logvar) + mu ** 2))
        return kld

    def return_loss(self, reconstructions, target, mu, logvar,beta=1):
        recon_loss = self.mse(reconstructions, target)
        #self.bce = nn.BCELoss(reduction='mean')
        #recon_loss = self.bce(reconstructions,target)
        kld_loss = self.KLD(mu, logvar)
        
     
        total_loss = recon_loss + (beta * kld_loss)

        return total_loss, recon_loss, kld_loss