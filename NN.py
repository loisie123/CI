import torch
from torch.autograd import Variable
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, forward_info):

        super(Net, self).__init__()

        # create module list
        self.activ = []
        for i in range(len(forward_info)-1):
            print(forward_info[i][0])
            if forward_info[i][0] == 'l':
                self.activ.append(nn.Linear(forward_info[1][i], forward_info[1][i+1]))
            elif forward_info[i][0] == 't':
                self.activ.append(nn.Tanh(forward_info[1][i], forward_info[1][i+1]))
            elif forward_info[i][0] == 's':
                self.activ.append(nn.Sigmoid(forward_info[1][i], forward_info[1][i+1]))
            else:
                raise ValueError("forward_info is not well-defined")


    def forward(self, x):

        # als dit niet goed werkt kan je andere opties hier beneden even proberen
        for activ_func in self.activ:
            x = activ_func(x)

        #x = self.activ[0](x)
        #for i in range(1,len(self.activ)):
            #x = self.activ[i](nn.functional.relu(x))

        #x = self.activ[0](x)
        #for i in range(1,len(self.activ)):
            #x = self.activ[i](x..clamp(min=0))

        return x
