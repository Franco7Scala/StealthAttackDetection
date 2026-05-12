import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, nc=16, embedding_dim=8):
        super().__init__()

        self.nc = nc
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(self.nc, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),

            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),

            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),

            nn.Linear(self.embedding_dim, self.embedding_dim)
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
        z = self.net(x)
        return z


class SiameseNetwork(nn.Module):
    def __init__(self, nc=16, embedding_dim=8):
        super().__init__()
        self.nc = nc
        self.embedding_dim = embedding_dim

        self.embedder = Embedder(nc=self.nc, embedding_dim=self.embedding_dim)

    def forward(self, x1, x2):
        z1 = self.embedder(x1)
        z2 = self.embedder(x2)

        return z1, z2


class Classifier(nn.Module):
    def __init__(self, embedding_dim=8,n_classes=1):
        super().__init__()
        self.embedding_dim = embedding_dim

        

        # Classification Head mutuato dalla SimpleMLP
        #self.classifier = nn.Sequential(
         #   nn.Linear(self.embedding_dim, self.embedding_dim//2),
         #   nn.ReLU(),
         #   nn.Linear(self.embedding_dim//2, n_classes) 
       # )

       # self.init_weights()

        
        self.fc = nn.Linear(embedding_dim, 1)

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
        return self.fc(x).flatten()
        #out = self.classifier(x)
        #return out.flatten()