from spaghettini import quick_register

from src.dl.models.deepsets.deepsets import DeepSetOriginal
from src.dl.models.gnns.gin import GIN
from src.dl.models.transformers.set_transformers import SetTransformer
from src.dl.models.transformers.encoders_decoders import SetTransformerEncoder, SetTransformerDecoder
from src.dl.models.gnns.nested_gin import NestedGIN
from src.dl.models.gnns.gin_residual import ResidualGIN


@quick_register
def get_default_deepset():
    return DeepSetOriginal(dim_input=841, num_outputs=1, dim_output=29, dim_hidden=1065)


@quick_register
def get_default_gin():
    return GIN(num_layers=6, num_mlp_layers=2, input_dim=841, hidden_dim=995, output_dim=29, final_dropout=0.,
               learn_eps=True, graph_pooling_type="mean", neighbor_pooling_type="sum")


@quick_register
def get_default_residual_gin():
    return ResidualGIN(num_layers=6, num_mlp_layers=2, input_dim=841, hidden_dim=995, output_dim=29, final_dropout=0.,
                       learn_eps=True, graph_pooling_type="mean", neighbor_pooling_type="sum")


@quick_register
def get_default_set_transformer():
    num_heads = 20
    dim_head = 28
    hid_dim = num_heads * dim_head
    encoder = SetTransformerEncoder(num_heads=num_heads, dim_elements=841, dim_hidden=hid_dim, dim_out=hid_dim,
                                    add_layer_norm=True, pre_ln=True)
    decoder = SetTransformerDecoder(num_heads=num_heads, num_seed_vectors=29, dim_elements=hid_dim, dim_hidden1=hid_dim,
                                    dim_hidden2=hid_dim, dim_out=1, add_layer_norm=True, pre_ln=True)
    return SetTransformer(encoder=encoder, decoder=decoder)


@quick_register
def get_nested_set_network():
    return NestedGIN(num_layers=10, input_dim=30, hidden_dim=480, output_dim=1, final_dropout=0., learn_eps=True,
                     neighbor_pooling_type="sum", graph_pooling_type="mean")
