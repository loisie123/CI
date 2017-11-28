import torch
from torch.autograd import Variable
import torch.nn as nn


# verandereringen zijn
## NN ipv Net
## layer_info = list van aantal nodes per hidden layer

class AttrProxy(object):

    """
        Translates index lookups into attribute lookups.
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class NN(nn.Module):

    """
        Foot forward neural network.
    """

    def __init__(self, layer_info):

        super(NN, self).__init__()

        if len(layer_info) < 3:
            raise ValueError("layer_info not wel-defined: list with at least three integers expected")

        # create modules
        self.steps = len(layer_info)-1
        for i in range(self.steps):
            self.add_module('step_' + str(i), nn.Linear(layer_info[i], layer_info[i+1]))
        self.step = AttrProxy(self, 'step_')

    def forward(self, x):
        tanh = nn.Tanh()
        x = self.step[0](x)
        for i in range(1, self.steps-1):
            x = self.step[i](tanh(x))
        x = self.step[self.steps-1](x)
        return x
