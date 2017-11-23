# Code in file autograd/two_layer_net_autograd.py
# first neural network.

import torch
from torch.autograd import Variable
import torch.nn as nn
import csv
from NN import *

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


def open_file(path_to_filename, path_to_filename2, path_to_filename3)
    with open(path_to_filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        X = []
        for row in readCSV:
            X.append(row)
    if path_to_filename2 != None:
        with open(path_to_filename2) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in readCSV:
                X.append(row)
    if path_to_filename3 != None:
        with open(path_to_filename3) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in readCSV:
                X.append(row)
    return X


def create_folds(X, k):
    """
    Create k folds of equal size for data set X.
    X is a list and k is a non-zero integer.
    """
    N = len(X)
    if k > N: k = N # set k to N if k if bigger than the size of data set
    fold_size = round(N/k)
    folds = []
    for i in range(k):
        print("Gaat dit goed?")
        train_set = X[:i*fold_size] + X[:(i+1)*fold_size]
        test_set = X[i*fold_size:(i+1)*fold_size]
        folds.append([train_set, test_set])
    return folds

                          
def train(fold, H, lr, iterations):
                          
    net = Net() # kloppen 3 en 23 ?
                          
    train_data = fold[0]
    train_input = Variable(torch.FloatTensor(train_data[:][3:]).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(train_data[:][:3]).type(dtype), requires_grad=False)
                                                    
    # try different loss functions
    loss_function = nn.MSELoss()
    # nn.CrossEntropyLoss()
    
    # try different weight optimizers
    # for example: SGD, Nesterov-SGD, Adam, RMSProp, etc
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
                          
    for i in range(iterations):
        # zero gradient buffers
        optimizer.zero_grad()
                          
        # find output of network
        train_output = net(train_input)
                          
        # error of output and target
        loss = loss_function(train_output, train_target)
                          
        # backpropagate the error                     
        loss.backward()
                          
        # update the weights
        optimizer.step()        
          
    
    # find loss of test set
    test_data = fold[1]
    test_input = Variable(torch.FloatTensor(test_data[:][3:]).type(dtype), requires_grad=False)
    test_target = Variable(torch.FloatTensor(test_data[:][:3]).type(dtype), requires_grad=False)
    test_output = net(test_input)
    test_loss = loss_function(test_output, test_target)
                          
    return net.parameters(), test_loss                   
                          
                          
"""
parameters:
path_to_filename
path_to_filename2 (default is None)
path_to_filename3 (default is None)
H: list of non-zero integers; the kth number corresponds with the number of nodes in the kth hidden layer
lr: learning rate (default is 1e-17)
iterations: number of iterations (non-zero integer)
k: number of folds (non-zero integer)
"""                    
def main(path_to_filename, path_to_filename2 = None, path_to_filename3  = None, H, lr = 1e-17, iterations, k):
                          
    X = open_file(path_to_filename, path_to_filename2, path_to_filename3)
    folds = create_folds(X, k)
                          
    best_weights = None        
    error = 10^40        
    for fold in folds:
        weights, test_loss = train(fold, H, lr, iterations)
        if error > test_loss:
            error = test_loss
            best_weights = weights
                          
    return best_weights
