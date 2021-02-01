from spaghettini import quick_register

import torch
import torch.nn as nn


@quick_register
class FullyConnected(nn.Module):
    def __init__(self, dim_input, dim_output, num_layers, dim_hidden=512, max_voter_num=99):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.max_voter_num = max_voter_num

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
        breakpoint()
        # Add dummy voters and flatten.
        bs, num_voter, cand_sq = X.shape
        X_full = torch.zeros((bs, self.max_voter_num, cand_sq)).type_as(X)
        X_full[:, :num_voter, :] = X
        X_full = X_full.view(bs, -1)

        # Run the network.
        return self.network(X_full)

    @property
    def name(self):
        return "FullyConnected"
