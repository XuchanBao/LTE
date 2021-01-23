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
                 activation=F.relu):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # input layer
                self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            else:
                # hidden layers
                for i in range(num_layers - 1):
                    self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

        # output layer
        self.linear_output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, g):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        return self.linear_output_layer(hg)
