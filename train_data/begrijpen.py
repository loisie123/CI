import numpy as np
import csv
import os

def sigmoid(x):
    return 1/(1-np.exp(-x))

def data():
    data = []
    in_data = []
    target = []
    with open('/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            in_data.append(row[4:25])
            target.append(row[0:3])
    return in_data[0:10], target[0:10]


#function that calculates a step forward.

#klopt iets net,
def forward(inp, W):
    b = 0.001
    z1 = inp * W[0]
    z2 = inp * W[1]
    z3 = inp * W[2]
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    a3 = sigmoid(z3)
    output = [sum(a1), sum(a2), sum(a3)]
    return output


def update_weights( target, W , delta):
    theta = 0.01

    for i in range(len(target)):
        W[i] += theta * delta[i]

    return W




def cost_function(output, target):
    error = np.zeros(len(output))

    for i in range(len(output)):
        out  = float(output[i])
        t = float(target[i])

        error[i] = (t - out)**2
    return error

def sigmoid_derivative(x):
    return x * (1 - x)


# trains the model
def train_NN(input, hidden_nodes, layers):
    # make weights. This is weight to one node.
    # we have 21
    W = []
    for i in range(hidden_nodes):
        W.append(np.random.random(input))
    print(W)
    #W is now weight matrix of input  x nodes

    in_data, target = data()

    #for iter1 in range(0,len(in_data)-2):
    for iter1 in range(len(in_data)):
        l0 = []
        for number in in_data[iter1]:
            l0.append(float(number))
        l0 = np.asarray(l0)

        output = forward(l0, W)
        error = cost_function(output, target[iter1])

        # calculate the derivative
        for i in range(len(error)):
             error[i]= sigmoid_derivative(float(error[i]))
        W= update_weights( target[i], W , error)


    Model_W = W
    print (Model_W)
# we need 3 outputs because we have 3 targets.

train_NN(21, 3, 1)  #  input is 21
