import torch
from torch.autograd import Variable
import torch.nn as nn
import csv
from NN import *

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


def train(net, fold, lr, iterations):

    # make data ready for use
    train_data = fold[0]
    in_data = []
    out_data = []
    print("Error? dan werkt row[3:]+[0]*36 op regel 58 niet")
    for row in train_data:
        #in_data.append(row[3:]+[0]*36)
        # in_data.append(row[3:]+[200]*36)
        # in_data.append(row[3:]+[0 for i in range(36)])
        in_data.append(row[3:])
        out_data.append(row[:3])
    train_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)

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
    in_data = []
    out_data = []
    for row in train_data:
        in_data.append(row[3:])
        out_data.append(row[:3])
    test_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    test_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)
    test_output = net(test_input)
    test_loss = loss_function(test_output, test_target)

    return net.parameters(), test_loss.data[0]


"""
parameters:
path_to_filename
path_to_filename2 (default is None)
path_to_filename3 (default is None)
lr: learning rate (default is 1e-17)
iterations: number of iterations (non-zero integer)
k: number of folds (non-zero integer)
"""
def main(net, iterations, k, path_to_filename, path_to_filename2 = None, path_to_filename3  = None,  lr = 1e-17):

    X = open_file(path_to_filename, path_to_filename2, path_to_filename3)
    folds = create_folds(X, k)

    best_weights = None
    error = 10^40
    for fold in folds:
        weights, test_loss = train(net, fold, lr, iterations)
        if error > test_loss:
            error = test_loss
            best_weights = weights

    # als het goed is werkt de k-fold cross validation nu wel
    # als het toch niet werkt dan kan je het weer commenten en de volgende regel uncommenten:
    # best_weights, _ = train(net, folds[0], lr, iterations)

    return best_weights


"""
Dingen die wij kunnen proberen zijn:
- forward_info aanpassen
- loss_function aanpassen (in train())
- optimizer aanpassen (in train())
- k aanpassen
- lr aanpassen -> veel gebruikte techniek is: start with large, if training crit div, try 3 times smaller, etc until no diverges is observed

"""

# list van tuples
# een tuple bevat een string: 'l'(=lineair), 's'(=sigmoid) of 't'(=tanh) die staat voor de soort activatiefunctie
# LET OP: de eerste string is leeg; deze gebruiken wij namelijk niet.. (je kan er dus ook iets anders leuks opschrijven)
# een tuple bevat een integer dat staat voor de aantal nodes in de desbetreffende layer

forward_info = [('l', 22), ('s', 8), ('t', 5), ('l', 3)]
#
net = Net(forward_info)

print(net.parameters())
#
main(net, 10000, 5, '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv',path_to_filename2 = '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/alpine-1.csv', path_to_filename3 = '//Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/f-speedway.csv' )
print(net.parameters())
