from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn


torch.manual_seed(42)

dtype = torch.FloatTensor


N = 64 # batch siz for crossvalidation
D_input = 1000 # amount of input variables
H = 100  # dimension of your hidden layer
D_out = 10 # output Dimension 


# over deze variabelen hoeven geen gradients berekent te worden. Dit zijn namelijk de input
# en output variabelen.
# input
X = Variable(torch.randn(N,D_input).type(dtype), requires_grad = False)
import sys
sys.stdout.buffer.write(chr(9986).encode('utf8'))





# output
Y = Variable(torch.randn(N,D_out).type(dtype), requires_grad = False)

# make weights
# from input to hiddenlayer:
w1 = Variable(torch.randn(D_input, H).type(dtype), requires_grad = True)
#weight from hiddenlayer tot output
w2 = Variable(torch.randn( H, D_out).type(dtype), requires_grad = True)


# define a learning rate.
learing_rate = 1e-6

# train neural network door 500 keer het volgende te doen.
for t in range(500):
    y_pred = X.mm(w1).clamp(min=0).mm(w2)

    #compute loss:
    loss = (y_pred - Y).pow(2).sum()


    #manually zero the gradients before running the backpars pass:
    w1.grad.data.zero_()
    w2.grad.data.zero_()

    #backward propegation:
    loss.backward()
    print(t, loss.data[0])


    #update weights:
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data
