"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from spaghettini import quick_register
from dgl.heterograph import DGLHeteroGraph

from src.dl.models.transformers.encoders_decoders import MultiHeadAttentionBlock


class ApplyNodeFunc(nn.Module):
    def __init__(self, fn):
        super(ApplyNodeFunc, self).__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(self.fn.output_dim)

    def forward(self, h):
        h = self.fn(h)
        h = self.ln(h)
        h = F.relu(h)
        return h


@quick_register
class SingleBlockSetTransformerEncoder(nn.Module):
    def __init__(self, num_heads, dim_elements, dim_out, add_layer_norm):
        super().__init__()
        self.sab = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_elements,
            dim_values=dim_elements,
            dim_transformed_keys_queries=dim_out // num_heads,
            dim_transformed_values=dim_out // num_heads,
            attention_layer_dim_out=dim_out,
            add_layer_norm=add_layer_norm)
        self.output_dim = dim_out

    def forward(self, xs):
        return self.sab(xs, xs, xs)


@quick_register
class NestedGIN(nn.Module):
    """GIN model"""

    def __init__(self, num_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(NestedGIN, self).__init__()
        assert final_dropout == 0., print("Don't use dropout for now. ")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.layernorms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                fn = SingleBlockSetTransformerEncoder(num_heads=4, dim_elements=input_dim, dim_out=hidden_dim,
                                                      add_layer_norm=True)
            else:
                fn = SingleBlockSetTransformerEncoder(num_heads=4, dim_elements=hidden_dim, dim_out=hidden_dim,
                                                      add_layer_norm=True)
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(fn=fn), neighbor_pooling_type, 0, self.learn_eps))
            self.layernorms.append(nn.LayerNorm(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, **kwargs):
        h = g.ndata['feat'] if isinstance(g, DGLHeteroGraph) else g

        # Reshape the tensor.
        bs_nv, dc = h.shape
        h = h.view(bs_nv, -1, self.input_dim)

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.layernorms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h).squeeze(-1))

        return score_over_layer

    @property
    def name(self):
        return "NestedGIN"


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.gnns.nested_gin
    """
    test_num = 0

    if test_num == 0:
        import random
        from src.data.datasets.ballot import Ballot
        # Check if NestedGIN is candidate-equivariant.
        blt = Ballot(max_num_candidates=20, one_hot_candidate_dim=20)
        xs, ys, utilities = blt[0]
        ng = NestedGIN(num_layers=5, input_dim=20, hidden_dim=512, output_dim=1, final_dropout=0., graph_pooling_type="sum", neighbor_pooling_type="sum", learn_eps=True)

        ys = ng(xs)

        # Permute the candidates.
        xst = xs.ndata['feat']
        xst = xst.view(-1, 20, 20)
        reordering = torch.tensor(random.sample(list(range(20)), k=20))
        xst_reordered = xst[:, reordering, :]

        # Check invariance.
        xs.ndata['feat'] = xst_reordered.view(-1, 400)
        ys_reordered = ng(xs)
        print(ys_reordered)
        print(ys[:, reordering])
        print(ys_reordered == ys[:, reordering])



