import math
import torch

from torch.nn.init import calculate_gain


def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)


def custom_xavier_uniform_(tensor, fan_in_dim=-1, fan_out_dim=-2, gain=1.):
    assert tensor.ndimension() == 3, "This is only tested for dim=3 tensors. "
    fan_in_num = tensor.shape[fan_in_dim]
    fan_out_num = tensor.shape[fan_out_dim]

    std = gain * math.sqrt(2.0 / float(fan_in_num + fan_out_num))
    a = math.sqrt(3.0) * std

    return _no_grad_uniform_(tensor, -a, a)


def custom_kaiming_uniform_(tensor,
                            fan_in_dim,
                            a=0.,
                            nonlinearity="leaky_relu"):
    fan_in_num = tensor.shape[fan_in_dim]
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in_num)
    bound = math.sqrt(3) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)