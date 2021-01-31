import math
import numpy as np

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, uniform_, kaiming_normal
from torch.nn.init import xavier_uniform_
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules import Linear

from src.dl.initialization.custom_initializers import custom_kaiming_uniform_


class SAB(nn.Module):
    def __init__(self,
                 num_heads,
                 dim_keys_queries,
                 dim_values,
                 dim_transformed_keys_queries,
                 dim_transformed_values,
                 attention_layer_dim_out,
                 add_layer_norm=False,
                 add_residual_tokens=True,
                 pre_ln=False):
        super().__init__()
        assert attention_layer_dim_out == dim_transformed_values * num_heads, print(
            'num_head doesnt divide dim out without reminder')
        # Initialize layers.
        self.multihead_attention_layer = MultiHeadAttentionLayer(
            num_heads=num_heads,
            dim_keys_queries=dim_keys_queries,
            dim_values=dim_values,
            dim_transformed_keys_queries=dim_transformed_keys_queries,
            dim_transformed_values=dim_transformed_values,
            dim_out=attention_layer_dim_out)

        self.rff1 = Linear(in_features=attention_layer_dim_out,
                           out_features=attention_layer_dim_out)
        self.rff2 = Linear(in_features=attention_layer_dim_out,
                           out_features=attention_layer_dim_out)
        if add_layer_norm:
            if pre_ln:
                self.norm1 = LayerNorm(dim_keys_queries)
            else:
                self.norm1 = LayerNorm(attention_layer_dim_out)
            self.norm2 = LayerNorm(attention_layer_dim_out)


        self.add_ln = add_layer_norm
        self.add_residual_tokens = add_residual_tokens
        self.pre_ln = pre_ln  # Whether to apply layernorm before the attention and residual or not.

    def forward(self, tokens):
        # Post layer norm version.
        if self.pre_ln is False:
            multihead_out, transformed_queries = self.multihead_attention_layer(
                queries=tokens, keys=tokens, values=tokens)
            if self.add_residual_tokens:
                # multihead_out = multihead_out + transformed_queries
                multihead_out = multihead_out + tokens

            h = self.norm1(multihead_out) if self.add_ln else multihead_out

            h = h + self.rff2(torch.relu(self.rff1(h)))

            output = self.norm2(h) if self.add_ln else h

            return output
        else:  # Pre-layer norm version.
            # Multihead attention part.
            pre_attention_tokens = self.norm1(tokens) if self.add_ln else tokens
            multihead_out, transformed_queries = self.multihead_attention_layer(
                queries=pre_attention_tokens, keys=pre_attention_tokens, values=pre_attention_tokens)
            if self.add_residual_tokens:
                # multihead_out = multihead_out + transformed_queries
                multihead_out = multihead_out + tokens

            # Fully connected part.
            pre_residual_tokens = self.norm2(multihead_out) if self.add_ln else multihead_out
            output = multihead_out + self.rff2(torch.relu(self.rff1(pre_residual_tokens)))

            return output


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 num_heads,
                 dim_keys_queries,
                 dim_values,
                 dim_transformed_keys_queries,
                 dim_transformed_values,
                 attention_layer_dim_out,
                 add_layer_norm=False,
                 add_residual_queries=True,
                 pre_ln=False):
        super().__init__()
        assert attention_layer_dim_out == dim_transformed_values * num_heads, print(
            'num_head doesnt divide dim out without reminder')
        # Initialize layers.
        self.multihead_attention_layer = MultiHeadAttentionLayer(
            num_heads=num_heads,
            dim_keys_queries=dim_keys_queries,
            dim_values=dim_values,
            dim_transformed_keys_queries=dim_transformed_keys_queries,
            dim_transformed_values=dim_transformed_values,
            dim_out=attention_layer_dim_out)

        self.rff1 = Linear(in_features=attention_layer_dim_out,
                           out_features=attention_layer_dim_out)
        self.rff2 = Linear(in_features=attention_layer_dim_out,
                           out_features=attention_layer_dim_out)
        if add_layer_norm:
            self.norm1 = LayerNorm(attention_layer_dim_out)
            self.norm2 = LayerNorm(attention_layer_dim_out)
            self.norm1q = LayerNorm(dim_keys_queries)
            self.norm1k = LayerNorm(dim_keys_queries)
            self.norm1v = LayerNorm(dim_keys_queries)

        self.add_ln = add_layer_norm
        self.add_residual_queries = add_residual_queries
        self.pre_ln = pre_ln

    def forward(self, queries, keys, values):
        # Post layer norm version.
        if self.pre_ln is False:
            multihead_out, transformed_queries = self.multihead_attention_layer(
                queries=queries, keys=keys, values=values)
            if self.add_residual_queries:
                # multihead_out = multihead_out + transformed_queries
                multihead_out = multihead_out + queries

            h = self.norm1(multihead_out) if self.add_ln else multihead_out

            h = h + self.rff2(torch.relu(self.rff1(h)))

            output = self.norm2(h) if self.add_ln else h

            return output
        else:  # Pre layer norm version.
            # Multihead attention part.
            pre_att_queries = self.norm1q(queries) if self.add_ln else queries
            pre_att_keys = self.norm1k(keys) if self.add_ln else keys
            pre_att_values = self.norm1v(values) if self.add_ln else keys
            multihead_out, transformed_queries = self.multihead_attention_layer(
                queries=pre_att_queries, keys=pre_att_keys, values=pre_att_values)
            if self.add_residual_queries:
                # multihead_out = multihead_out + transformed_queries
                multihead_out = multihead_out + queries

            # Fully connected part.
            pre_residual_tokens = self.norm2(multihead_out) if self.add_ln else multihead_out
            output = multihead_out + self.rff2(torch.relu(self.rff1(pre_residual_tokens)))

            return output


class PMA(nn.Module):
    def __init__(self, num_heads, num_seed_vectors, dim_elements, dim_out, add_layer_norm, pre_ln=False):
        super().__init__()
        assert dim_out == dim_elements, print(
            "Different output shape not supported yet")

        # Initialize the learnable seed vectors.
        self.S = Parameter(torch.Tensor(1, num_seed_vectors, dim_elements))
        self._initialize_seed_vectors()

        # Initialize the attention block.
        self.attention_block = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_elements,
            dim_values=dim_elements,
            dim_transformed_keys_queries=dim_elements // num_heads,
            dim_transformed_values=dim_elements // num_heads,
            attention_layer_dim_out=dim_elements,
            add_layer_norm=add_layer_norm,
            pre_ln=pre_ln)

    def _initialize_seed_vectors(self):
        xavier_uniform_(self.S)

    def forward(self, xs):
        bs = xs.shape[0]
        #         transformed_xs = self.rff(xs)
        return self.attention_block(self.S.repeat((bs, 1, 1)), xs, xs)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 num_heads,
                 dim_keys_queries,
                 dim_values,
                 dim_transformed_keys_queries,
                 dim_transformed_values,
                 dim_out,
                 add_residual_query_connection=True):
        super().__init__()
        assert dim_transformed_values > 0 and dim_transformed_keys_queries > 0
        self.num_heads = num_heads
        self.dim_kq = dim_keys_queries
        self.dim_v = dim_values
        self.dim_tkq = dim_transformed_keys_queries
        self.dim_tv = dim_transformed_values
        self.dim_out = dim_out
        self.add_residual_query_connection = add_residual_query_connection

        # Input transformation parameters.
        self.Wq_in = Parameter(
            torch.Tensor(self.num_heads, self.dim_kq, self.dim_tkq))
        self.bq_in = Parameter(torch.Tensor(self.num_heads, 1, self.dim_tkq))
        self.Wk_in = Parameter(
            torch.Tensor(self.num_heads, self.dim_kq, self.dim_tkq))
        self.bk_in = Parameter(torch.Tensor(self.num_heads, 1, self.dim_tkq))
        self.Wv_in = Parameter(
            torch.Tensor(self.num_heads, self.dim_v, self.dim_tv))
        self.bv_in = Parameter(torch.Tensor(self.num_heads, 1, self.dim_tv))

        # Initialize the parameters.
        self._initialize_parameters()

    def _initialize_parameters(self):
        custom_kaiming_uniform_(self.Wq_in, fan_in_dim=1, a=math.sqrt(5))
        custom_kaiming_uniform_(self.Wk_in, fan_in_dim=1, a=math.sqrt(5))
        custom_kaiming_uniform_(self.Wv_in, fan_in_dim=1, a=math.sqrt(5))

        # The following replicates Kaiming initialization for biases.
        bias_kq_bound = 1 / math.sqrt(self.dim_kq)
        bias_v_bound = 1 / math.sqrt(self.dim_v)
        uniform_(self.bq_in, -bias_kq_bound, bias_kq_bound)
        uniform_(self.bk_in, -bias_kq_bound, bias_kq_bound)
        uniform_(self.bv_in, -bias_v_bound, bias_v_bound)

    def forward(self, queries, keys, values):
        # Dimensions should be : (batch_size, num_vectors, dim)
        assert queries.ndimension() == keys.ndimension() == values.ndimension(
        ) == 3
        bs = max(queries.shape[0], keys.shape[0],
                 values.shape[0])  # batch_size
        num_queries = queries.shape[1]

        # Transform the queries, keys and values.
        transformed_queries = queries.unsqueeze(1) @ self.Wq_in + self.bq_in
        transformed_keys = keys.unsqueeze(1) @ self.Wk_in + self.bk_in
        transformed_values = values.unsqueeze(1) @ self.Wv_in + self.bv_in

        # Perform parallelized single head attention on the transformed queries, keys and values.
        out_values = parallelized_single_head_attention(
            queries=transformed_queries,
            keys=transformed_keys,
            values=transformed_values)

        # Transform the output.
        out_values = out_values.transpose(1, 2).transpose(2, 3).reshape(
            (bs, num_queries, self.dim_tv * self.num_heads))

        transformed_queries = transformed_queries.transpose(1, 2).reshape(
            bs, num_queries, self.dim_tv * self.num_heads)

        return out_values, transformed_queries


def parallelized_single_head_attention(queries, keys, values):
    # Do some checks to make sure the dimensionalities are consistent.
    assert queries.ndimension() == keys.ndimension() == values.ndimension(
    ) == 4, "Wrong input dimensions. "
    assert queries.shape[1] == keys.shape[1] == values.shape[1], print(
        "Head dimensions don't match. ")
    assert queries.shape[3] == keys.shape[3], print(
        "Query-key vector dimensions don't match. ")
    assert keys.shape[2] == values.shape[2], print(
        "Number of keys and values should match. ")

    # Compute similarities.
    query_key_similarities = queries @ keys.transpose(2, 3)

    # Normalize the similarities.
    scaling_factor = math.sqrt(values.shape[1] * values.shape[3])  # heads * dim_values
    normalized_similarities = scaled_softmax(query_key_similarities,
                                             scaling_factor,
                                             dim=3)

    # Take the weighed average of the values.
    weighted_values = normalized_similarities @ values

    return weighted_values


def scaled_softmax(tensor, scaling_factor, dim):
    return torch.softmax((tensor / scaling_factor), dim=dim)
