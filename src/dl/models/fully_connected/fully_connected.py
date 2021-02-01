from spaghettini import quick_register

import torch.nn as nn


@quick_register
class FullyConnected(nn.Module):
    def __init__(self, dim_input, dim_output, num_layers, dim_hidden=512):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        layers = list()
        # First layer.
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(dim_hidden))
        # Middle layers.
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.LeakyReLU())
            layers.append(nn.LayerNorm(dim_hidden))
        # Final layer.
        layers.append(nn.Linear(dim_hidden, dim_output))

        self.network = nn.Sequential(*layers)

    def forward(self, X, **kwargs):
        bs = X.shape[0]
        X = X.view(bs, -1)
        return self.network(X)

    @property
    def name(self):
        return "FullyConnected"
