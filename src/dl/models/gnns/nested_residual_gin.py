"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from spaghettini import quick_register
from dgl.heterograph import DGLHeteroGraph


from src.dl.models.transformers.encoders_decoders import SAB


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, fn):
        super(ApplyNodeFunc, self).__init__()
        self.fn = fn
        # self.bn = nn.BatchNorm1d(self.mlp.output_dim)
        # self.ln = nn.LayerNorm(self.mlp.input_dim)

    def forward(self, h):
        # h = self.ln(h)
        h = self.fn(h)
        return h


@quick_register
class SABBlock(nn.Module):
    def __init__(self, num_heads, dim_elements, dim_out, add_layer_norm, pre_ln=True, add_residual_tokens=True):
        super().__init__()
        self.sab = SAB(
            num_heads=num_heads,
            dim_keys_queries=dim_elements,
            dim_values=dim_elements,
            dim_transformed_keys_queries=dim_out // num_heads,
            dim_transformed_values=dim_out // num_heads,
            attention_layer_dim_out=dim_out,
            add_layer_norm=add_layer_norm,
            pre_ln=pre_ln,
            add_residual_tokens=add_residual_tokens)
        self.output_dim = dim_out

    def forward(self, xs):
        return self.sab(xs)


@quick_register
class NestedResidualGIN(nn.Module):
    """GIN model"""

    def __init__(self, num_layers, input_dim, num_heads, dim_per_head,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.hidden_dim = num_heads * dim_per_head

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()
        self.layer_norms_1 = torch.nn.ModuleList()
        self.layer_norms_2 = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                fn = SABBlock(num_heads=num_heads, dim_elements=input_dim, dim_out=self.hidden_dim,
                              add_layer_norm=True, pre_ln=True, add_residual_tokens=False)
            else:
                fn = SABBlock(num_heads=num_heads, dim_elements=self.hidden_dim, dim_out=self.hidden_dim,
                              add_layer_norm=True, pre_ln=True)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(fn), neighbor_pooling_type, 0, self.learn_eps))
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layer_norms_1.append(nn.LayerNorm(self.hidden_dim))
            self.layer_norms_2.append(nn.LayerNorm(self.hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(self.hidden_dim, output_dim))

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
            if i == 0:
                h = self.ginlayers[i](g, h)
            else:
                h_pre_gin_layer = self.layer_norms_1[i](h)
                h = h + self.ginlayers[i](g, h_pre_gin_layer)
            # h = self.layer_norms[i](h)
            # h = F.relu(h)
            hidden_rep.append(self.layer_norms_2[i](F.relu(h)))

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer

    @property
    def name(self):
        return "NestedResidualGIN"


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.gnns.nested_residual_gin
    """
    test_num = 0

    if test_num == 0:
        from src.data import get_default_mimicking_loader
        from src.voting.voting_rules import get_borda

        # Pass input through residual GIN.
        loader = get_default_mimicking_loader(distribution="uniform", voting_rule=get_borda(), return_graph=True)
        dataset = loader.dataset
        g, _, _ = dataset[0]
        nrg = NestedResidualGIN(num_layers=6, input_dim=841, num_heads=20, dim_per_head=28, output_dim=29,
                                final_dropout=0., learn_eps=False, graph_pooling_type="mean",
                                neighbor_pooling_type="sum")
        print(nrg)
        ys = nrg(g)
