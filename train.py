import torch
from torch.autograd import Variable
import torch.nn as nn
import csv
from NN import *
import random


dtype = torch.FloatTensor


def open_file(path_to_filename, path_to_filename2, path_to_filename3):
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
        train_set = X[:i*fold_size] + X[:(i+1)*fold_size]
        test_set = X[i*fold_size:(i+1)*fold_size]
        folds.append([train_set, test_set])
    return folds


def train(train_data, lr, iterations, layer_info):
    """
    Create neural netwerk and train it.
    Input:
        :param train_data
        :param lr
        :param iterations:
        :layer_info
    Output:
        neural netwerk
    """
    net = NN(layer_info)

    # make data ready for use
    in_data = []
    out_data = []
    for row in train_data:
        in_data.append(row[3:])
        out_data.append(row[:3])
    train_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)

    # define loss function: mean squared error
    loss_function = nn.MSELoss()

    # define optimize method: stochastic gradient descent
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

    return net


def test(test_data, net):
    """
        Find loss of test set.
    """
    # make data ready for use
    in_data = []
    out_data = []
    for row in test_data:
        in_data.append(row[3:])
        out_data.append(row[:3])
    test_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    test_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)

    # calculate output
    test_output = net(test_input)

    # define loss function: mean squared error
    loss_function = nn.MSELoss()

    # calculate the error
    test_loss = loss_function(test_output, test_target)

    return test_loss.data[0]


def create_nn(iterations, layer_info, path_to_filename, path_to_filename2=None, path_to_filename3=None, k=None, lr=1e-17):
    """
    Create a neural netwerk.
    Input:
        :param iterations: number of iterations (non-zero integer)
        :param k: number of folds (non-zero integer)
        :param layer_info: list of integers where element is the number of nodes of corresponding layer
        :param path_to_filename
        :param path_to_filename2 (default is None)
        :param path_to_filename3 (default is None)
        :param lr: learning rate (default is 1e-17)
    Output: neural netwerk
    """
    # create dataset from csv files
    X = open_file(path_to_filename, path_to_filename2, path_to_filename3)

    # not using k-fold cross validation
    if k == None or k <= 1:
        best_net = train(X, lr, iterations, layer_info)

    # using k-fold cross validation
    else:
        folds = create_folds(X, k)
        best_net = None
        error = None
        folds_prime = random.sample(folds, 2)
        for fold in folds_prime:
            net = train(fold[0], lr, iterations, layer_info)
            test_loss = test(fold[1], net)
            if error == None or error > test_loss:
                error = test_loss
                best_net = net

    return best_net







# AANROEPEN MET:
# create_nn(1000, [22,5,3], '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv',path_to_filename2 = '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/alpine-1.csv', path_to_filename3 = '//Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/f-speedway.csv')


# OF AANROEPEN MET k-fold cross validation: (hij zal maar met twee fold dingen doen)
# create_nn(1000, [22,5,3], '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv',path_to_filename2 = '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/alpine-1.csv', path_to_filename3 = '//Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/f-speedway.csv', k=20)
