import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, nc=16, nout=8, n_classes=1):
        super(SimpleMLP, self).__init__()
        self.nc = nc
        self.nout = nout
        
        # Feature Extractor (Hidden Layers)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.nc, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.ReLU(),

        )

        # Classification Head
        self.classifier = nn.Sequential(
            #nn.Linear(self.nout, self.nout // 2),
            #nn.ReLU(),
            nn.Linear(self.nout, n_classes) 
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight) 
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.flatten()