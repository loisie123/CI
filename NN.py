import torch
from torch.autograd import Variable
import torch.nn as nn
import random


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        numbers = [3,4,5,6,7,8]
        num_of_hidden_layers = random.choice(numbers)
        layer_info = [22] # [58]
        for i in range(num_of_hidden_layers):
            layer_info.append(random.choice(numbers))
        layer_info.append(3)

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
