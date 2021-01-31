from spaghettini import quick_register

from torch import nn
from torch.nn.modules import Linear

from src.dl.models.transformers.multihead_attention import MultiHeadAttentionBlock, PMA


@quick_register
class SetTransformerEncoder(nn.Module):
    def __init__(self, num_heads, dim_elements, dim_hidden, dim_out, add_layer_norm):
        super().__init__()
        self.sab1 = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_elements,
            dim_values=dim_elements,
            dim_transformed_keys_queries=dim_hidden // num_heads,
            dim_transformed_values=dim_hidden // num_heads,
            attention_layer_dim_out=dim_hidden,
            add_layer_norm=add_layer_norm,
            add_residual_queries=False)

        self.sab2 = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_hidden,
            dim_values=dim_hidden,
            dim_transformed_keys_queries=dim_hidden // num_heads,
            dim_transformed_values=dim_hidden // num_heads,
            attention_layer_dim_out=dim_hidden,
            add_layer_norm=add_layer_norm)

        self.sab3 = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_hidden,
            dim_values=dim_hidden,
            dim_transformed_keys_queries=dim_hidden // num_heads,
            dim_transformed_values=dim_hidden // num_heads,
            attention_layer_dim_out=dim_hidden,
            add_layer_norm=add_layer_norm)

        self.sab4 = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_hidden,
            dim_values=dim_hidden,
            dim_transformed_keys_queries=dim_hidden // num_heads,
            dim_transformed_values=dim_hidden // num_heads,
            attention_layer_dim_out=dim_out,
            add_layer_norm=add_layer_norm)

    def forward(self, xs):
        zs = self.sab1(xs, xs, xs)
        zs = self.sab2(zs, zs, zs)
        zs = self.sab3(zs, zs, zs)
        zs = self.sab4(zs, zs, zs)

        return zs


@quick_register
class SetTransformerDecoder(nn.Module):
    def __init__(self, num_heads, num_seed_vectors, dim_elements, dim_hidden1,
                 dim_hidden2, dim_out, add_layer_norm):
        super().__init__()
        self.pma = PMA(num_heads=num_heads,
                       num_seed_vectors=num_seed_vectors,
                       dim_elements=dim_elements,
                       dim_out=dim_hidden1,
                       add_layer_norm=add_layer_norm)
        self.sab = MultiHeadAttentionBlock(
            num_heads=num_heads,
            dim_keys_queries=dim_hidden1,
            dim_values=dim_hidden1,
            dim_transformed_keys_queries=dim_hidden2 // num_heads,
            dim_transformed_values=dim_hidden2 // num_heads,
            attention_layer_dim_out=dim_hidden2,
            add_layer_norm=add_layer_norm)
        self.rff = Linear(in_features=dim_hidden2, out_features=dim_out)

    def forward(self, xs):
        pma_out = self.pma(xs)
        return self.rff(self.sab(pma_out, pma_out, pma_out))


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.transformers.encoders_decoders
    """
    test_num = 0

    if test_num == 0:
        breakpoint()
        import torch
        # Pass random input through encoder.
        xs = torch.rand(size=(5, 20, 128))
        enc = SetTransformerEncoder(4, 128, 512, 512, add_layer_norm=True)

        ys = enc(xs)
