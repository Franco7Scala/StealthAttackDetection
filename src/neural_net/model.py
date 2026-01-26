import torch
import torch.nn as nn

class SimpleDiscriminator(nn.Module):

    def __init__(self, nc=121, hidden=16, nc_out=16):
        super(SimpleDiscriminator, self).__init__()


        self.feature_extractor = nn.Sequential(
            nn.Linear(nc, hidden),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.2),


            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, track_running_stats=False),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.2),

        )


        # output layer
        self.fc2 = nn.Sequential(
            nn.Linear(nc_out, 1),
            #nn.Sigmoid()
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
        noise =  torch.randn(x.size()).to(x.device) * 0.05 + 0.05
        x = x + noise
        x = self.feature_extractor(x)
        x = self.fc2(x)
        return x.flatten()


