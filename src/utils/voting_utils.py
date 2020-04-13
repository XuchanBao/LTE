import numpy as np
import torch


def get_one_hot(winner, num_candidates):
    if isinstance(winner, torch.Tensor):
        return (winner[..., None] == torch.arange(num_candidates)[None, ...]).type_as(winner)
    else:
        return winner[..., None] == np.arange(num_candidates)[None, ...]
