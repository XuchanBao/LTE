from spaghettini import quick_register

import numpy as np

from src.dl.models.deepsets.deepsets import DeepSetOriginal
from src.dl.models.gnns.gin import GIN
from src.dl.models.transformers.set_transformers import SetTransformer
from src.dl.models.transformers.encoders_decoders import SetTransformerEncoder, SetTransformerDecoder
from src.dl.models.gnns.nested_gin import NestedGIN
from src.dl.models.gnns.gin_residual import ResidualGIN
from src.dl.models.gnns.nested_residual_gin import NestedResidualGIN
from src.dl.models.fully_connected.fully_connected import FullyConnected


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
def get_default_nested_set_network():
    return NestedResidualGIN(num_layers=10, input_dim=29, num_heads=10, dim_per_head=16, output_dim=1, final_dropout=0.,
                             learn_eps=True, neighbor_pooling_type="sum", graph_pooling_type="mean")


@quick_register
def get_default_fully_connected():
    num_voters = 99
    num_cand = 29
    return FullyConnected(dim_input=num_voters*num_cand*num_cand, dim_output=29, num_layers=5, dim_hidden=237)


def get_default_small_fully_connected():
    num_voters = 99
    num_cand = 29
    return FullyConnected(dim_input=num_voters*num_cand*num_cand, dim_output=29, num_layers=5, dim_hidden=120)


def _num_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == "__main__":
    """
    Run from root. 
    python -m src.dl.models.defaults
    """
    test_num = 0

    if test_num == 0:
        # Print the number of parameters of the default models.
        ds = get_default_deepset()
        num_params_ds = _num_trainable_params(ds)
        print(f"Deepset has {num_params_ds:,} trainable params. ")

        gin = get_default_gin()
        num_params_gin = _num_trainable_params(gin)
        print(f"GIN has {num_params_gin:,} trainable params. ")

        st = get_default_set_transformer()
        num_params_st = _num_trainable_params(st)
        print(f"Set transformer has {num_params_st:,} trainable params. ")

        fc = get_default_fully_connected()
        num_params_fc = _num_trainable_params(fc)
        print(f"Fully con. net. has {num_params_fc:,} trainable params. ")

        fcs = get_default_small_fully_connected()
        num_params_fcs = _num_trainable_params(fcs)
        print(f"Small fully con. net. has {num_params_fcs:,} trainable params. ")





