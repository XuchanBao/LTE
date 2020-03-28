import numpy as np
import torch
from torch.utils.data import Dataset
from spaghettini import quick_register


@quick_register
class FindMax(Dataset):
    def __init__(self,
                 max_value=100,
                 max_len=10,
                 num_pad_zeros=0,
                 batch_size=1024,
                 epoch_length=100):
        self.max_value = max_value
        self.max_len = max_len
        self.num_pad_zeros = num_pad_zeros
        self.batch_size = batch_size
        self.epoch_length = epoch_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        length = np.random.randint(1, self.max_len + 1)
        x = np.random.randint(1, self.max_value, (self.batch_size, length, 1))

        if self.num_pad_zeros > 0:
            (bs, length, _) = x.shape
            new_x = np.zeros((bs, length, self.num_pad_zeros + 1))
            new_x[:, :, 0:1] = x
            x = new_x

        y = np.max(x, axis=(1, 2))[..., None, None]

        x_torch = torch.tensor(x).float()
        y_torch = torch.tensor(y).float()

        return x_torch, y_torch
