from spaghettini import quick_register

from src.dl.models.deepsets.deepsets import DeepSetOriginal
from src.dl.models.gnns.gin import GIN


@quick_register
def get_default_deepset():
    return DeepSetOriginal(dim_input=2500, num_outputs=1, dim_output=50, dim_hidden=970)


@quick_register
def get_default_gin():
    return GIN(num_layers=6, num_mlp_layers=2, input_dim=2500, hidden_dim=915, output_dim=50, final_dropout=0.,
               learn_eps=True, graph_pooling_type="mean", neighbor_pooling_type="sum")




