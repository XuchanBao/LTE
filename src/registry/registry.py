from spaghettini import register, quick_register, load, check
from munch import Munch

from torch.optim import Adam, SGD, RMSprop
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from torchvision.datasets import MNIST
from torchvision.datasets import DatasetFolder
from torch.nn.functional import mse_loss
from torch.nn import L1Loss, CrossEntropyLoss
from torch.nn.functional import relu, softplus
from torch import sigmoid, tanh
from torch.optim.lr_scheduler import StepLR


# Register basic.
quick_register(Munch)

# ____Pytorch Related____ #
# Data related.
quick_register(MNIST)
quick_register(DataLoader)
quick_register(DatasetFolder)
quick_register(ToTensor)
quick_register(Normalize)
quick_register(Compose)

# Optimizer related.
quick_register(Adam)
quick_register(SGD)
quick_register(RMSprop)
quick_register(StepLR)

# Losses.
quick_register(CrossEntropyLoss)
quick_register(mse_loss)
quick_register(L1Loss)

# Activations.
quick_register(relu)
quick_register(softplus)
quick_register(sigmoid)
quick_register(tanh)

# ____Pytorch Lightning Related___ #
quick_register(Experiment)
quick_register(Trainer)
quick_register(TestTubeLogger)
