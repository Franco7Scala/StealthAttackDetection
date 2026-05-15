import torch.nn as nn


class ProtoModel(nn.Module):


    def __init__(self, input_dim, output_dim, hidden_dims=[8, 8, 8]):
        super(ProtoModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward(self, x):
        return 0, self.network(x), 0, 0
