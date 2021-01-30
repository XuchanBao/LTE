from spaghettini import quick_register

from src.dl.models.deepsets.deepsets import DeepSetOriginal


def get_default_deepset():
    return DeepSetOriginal(dim_input=2500, num_outputs=1, dim_output=50, dim_hidden=970)


