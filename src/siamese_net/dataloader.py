import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicPairDataset(Dataset):
    def __init__(self, x, y, normal_samples_per_epoch=10):

        self.x = x
        self.y = y

        self.normal_idx = np.where(self.y == 0)[0]
        self.anomaly_idx = np.where(self.y == 1)[0]

        self.normal_samples_per_epoch = normal_samples_per_epoch

        self.pairs = []

        self.resample_pairs()

    def resample_pairs(self):

        self.pairs = []

        sampled_normals = np.random.choice(self.normal_idx, size=self.normal_samples_per_epoch, replace=False)

        for i, anchor_idx in enumerate(self.anomaly_idx):
            for other_anomaly_idx in self.anomaly_idx[i + 1:]:
                if anchor_idx == other_anomaly_idx:
                    continue

                self.pairs.append((anchor_idx, other_anomaly_idx, 1.0))

        for anchor_idx in self.anomaly_idx:
            for normal_idx in sampled_normals:

                self.pairs.append((anchor_idx, normal_idx, 0.0) )

        np.random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        x1 = self.x[idx1]
        x2 = self.x[idx2]
        label = torch.tensor(label)

        return x1, x2, label