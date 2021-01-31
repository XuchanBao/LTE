from spaghettini import quick_register

from src.dl.models.deepsets.deepsets import DeepSetOriginal
from src.dl.models.gnns.gin import GIN
from src.dl.models.transformers.set_transformers import SetTransformer
from src.dl.models.transformers.encoders_decoders import SetTransformerEncoder, SetTransformerDecoder
from src.dl.models.gnns.nested_gin import NestedGIN


@quick_register
def get_default_deepset():
    return DeepSetOriginal(dim_input=841, num_outputs=1, dim_output=29, dim_hidden=1065)


@quick_register
def get_default_gin():
    return GIN(num_layers=6, num_mlp_layers=2, input_dim=2500, hidden_dim=915, output_dim=50, final_dropout=0.,
               learn_eps=True, graph_pooling_type="mean", neighbor_pooling_type="sum")


@quick_register
def get_default_set_transformer():
    encoder = SetTransformerEncoder(num_heads=15, dim_elements=2500, dim_hidden=480, dim_out=480, add_layer_norm=True)
    decoder = SetTransformerDecoder(num_heads=15, num_seed_vectors=50, dim_elements=480, dim_hidden1=480,
                                    dim_hidden2=480, dim_out=1, add_layer_norm=True)
    return SetTransformer(encoder=encoder, decoder=decoder)


@quick_register
def get_nested_set_network():
    return NestedGIN(num_layers=10, input_dim=50, hidden_dim=480, output_dim=1, final_dropout=0., learn_eps=True,
                     neighbor_pooling_type="sum", graph_pooling_type="mean")
