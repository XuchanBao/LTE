from spaghettini import quick_register
import numpy as np


@quick_register
def load_numpy_array():
    def load_npy(path):
        return np.load(path)

    return load_npy
