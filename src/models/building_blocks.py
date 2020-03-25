from spaghettini import quick_register
from torch import nn
import torch.nn.functional as F


@quick_register
class SmallTwoConvTwoFC(nn.Module):
    def __init__(self, in_channels=1, num_fc_hidden=500, out_features=160, activation=F.relu, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, num_fc_hidden)
        self.bn1 = nn.BatchNorm1d(num_fc_hidden)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(num_fc_hidden, out_features)
        self.activation = activation

    def forward(self, x):
        assert x.ndimension() == 4
        batch_size, num_channels, height, width = x.shape
        z = self.activation(self.conv1(x))
        z = F.max_pool2d(z, 2, 2)
        z = self.activation(self.conv2(z))
        z = F.max_pool2d(z, 2, 2)
        z = z.view(batch_size, 5 * 5 * 50)
        z = self.activation(self.fc1(z))
        z = self.dropout1(z)
        z = self.fc2(z)

        return z


@quick_register
class SpecTwoConvTwoFC(nn.Module):
    def __init__(self, in_channels=1, num_fc_hidden=500, out_features=160, activation=F.relu, drop_prob=0.0):
        super().__init__()
        self.upsample = nn.Upsample(size=(140, 140,))
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 30, 5, 1)
        self.conv3 = nn.Conv2d(30, 40, 5, 1)
        self.conv4 = nn.Conv2d(40, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, num_fc_hidden)
        self.bn1 = nn.BatchNorm1d(num_fc_hidden)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(num_fc_hidden, out_features)
        self.activation = activation

    def forward(self, x):
        assert x.ndimension() == 4
        batch_size = x.shape[0]
        z = self.upsample(x)

        z = self.activation(self.conv1(z))
        z = F.max_pool2d(z, 2, 2)

        z = self.activation(self.conv2(z))
        z = F.max_pool2d(z, 2, 2)

        z = self.activation(self.conv3(z))
        z = F.max_pool2d(z, 2, 2)

        z = self.activation(self.conv4(z))
        z = F.max_pool2d(z, 2, 2)

        z = z.view(batch_size, 5 * 5 * 50)
        z = self.activation(self.fc1(z))
        z = self.dropout1(z)
        z = self.fc2(z)

        return z


@quick_register
class TwoLayerFC(nn.Module):
    def __init__(self, num_inputs=256, num_hidden=1000, num_outputs=784, activation=F.relu,
                 final_activation=lambda x: x):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation = activation
        self.final_activation = final_activation

        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_outputs)

    def forward(self, x):
        z = self.fc1(x)
        z = self.activation(z)
        z = self.fc2(z)
        z = self.final_activation(z)

        return z
