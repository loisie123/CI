import torch
from torch.autograd import Variable
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):

        #if len(H) == []: raise ValueError("list of layers is empty")

        super(Net, self).__init__()
        self.lin1 = nn.Tanh(22, 5)
        self.lin2 = nn.Linear(5, 3)
        # create module list
        #self.linear = [nn.Linear(D_in, H[0])]
        #for i in range(len(H)-1):
        #    self.linear.append(nn.Linear(H[i], H[i+1]))
        #self.linear.append(nn.Linear(H[-1],D_out))


    def forward(self, x):
        x = self.lin2(nn.functional.relu(self.lin1(x)))
        # of probeer
        # x = self.lin2(self.lin1(x).clamp(min=0))
        # x = self.lin2(self.lin1(x))

        # Max pooling over a (2, 2) window
        #x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        #x = nn.functional.relu(self.fc1(x))
        #x = nn.functional.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
