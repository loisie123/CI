# Code in file autograd/two_layer_net_autograd.py
import torch
from torch.autograd import Variable
import csv

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

with open('./train_data/aalborg.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)


    X = []
    for row in readCSV:
        X.append(row)

    del X[0]


# The in-data are all the external variables
in_data = []
for row in X:
    in_data.append(row[4:25])

# This is the output data (accelerate/brake/steering)
out_data = []
for row in X:
    out_data.append(row[0:3])

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4320, 21, 10, 3

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)


# x = Variable(torch.FloatTensor(out_data[0:100]).type(dtype), requires_grad=False)
# y = Variable(torch.FloatTensor(in_data[0:100]).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y using operations on Variables; these
  # are exactly the same operations we used to compute the forward pass using
  # Tensors, but we do not need to keep references to intermediate values since
  # we are not implementing the backward pass by hand.
  y_pred = x.mm(w1).clamp(min=0).mm(w2)

  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Variables with requires_grad=True.
  # After this call w1.grad and w2.grad will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients after running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

print(w1)
print(w2)