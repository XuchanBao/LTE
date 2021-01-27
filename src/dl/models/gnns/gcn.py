"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from spaghettini import quick_register


@quick_register
class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 dropout,
                 activation=F.relu,
                 residual_readout=False):
        super(GCN, self).__init__()
        self.residual_readout = residual_readout
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # input layer
                self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            else:
                # hidden layers
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

        # output layer
        if self.residual_readout:
            self.linear_output_layers = nn.ModuleList()
            for i in range(num_layers + 1):
                if i == 0:
                    self.linear_output_layers.append(nn.Linear(input_dim, output_dim))
                else:
                    self.linear_output_layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.linear_output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, g):
        h = g.ndata['feat']

        hidden_rep = [h]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            hidden_rep.append(h)
            g.ndata[f'h{i + 1}'] = h

        if self.residual_readout:
            score_over_layer = 0
            for i, hidden in enumerate(hidden_rep):
                if i == 0:
                    hidden_pooled = dgl.mean_nodes(g, 'feat')
                else:
                    hidden_pooled = dgl.mean_nodes(g, f'h{i}')
                score_over_layer += self.linear_output_layers[i](hidden_pooled)
        else:
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            score_over_layer = self.linear_output_layer(hg)
        return score_over_layer
