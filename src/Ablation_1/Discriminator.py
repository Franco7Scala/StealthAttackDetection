import torch
import numpy as np
import torch.nn as nn
import os
from typing import Optional
from tqdm import tqdm
class Discriminator(nn.Module):

    def __init__(self, nc=121, nc_out=16, nout=128):
        super(Discriminator, self).__init__()

        self.nc = nc
        self.nc_out = nc_out
        self.nout = nout

        self.feature_extractor = nn.Sequential(
            # features extractor
            nn.Linear(self.nc, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.ReLU(),

            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.ReLU(),

            # nn.Linear(self.nout * 2, self.nout * 4),
            # nn.BatchNorm1d(self.nout * 4, track_running_stats = False),
            # nn.LeakyReLU(0.2),
        )

        self.fc1 = nn.Sequential(
            # classifier
            nn.Linear(self.nout, self.nout),
            nn.BatchNorm1d(self.nout),
            nn.Dropout(0.05),
            nn.ReLU(),

            # nn.Linear(self.nout, self.nc_out * 4),
            # nn.BatchNorm1d(self.nc_out * 4, track_running_stats = False),
            # nn.ReLU(),

            # nn.Dropout(0.2),
            # nn.Linear(self.nc_out * 2, self.nc_out * 2),
            # nn.ReLU(),

            nn.Linear(self.nout, self.nc_out),
            nn.BatchNorm1d(self.nc_out),
            nn.Dropout(0.05),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.nc_out, 1),
            nn.Sigmoid()
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
        x = self.feature_extractor(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.flatten()

    def encode(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x