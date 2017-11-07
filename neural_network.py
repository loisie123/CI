import numpy as np
import csv

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
with open('./train_data/aalborg.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')


    X = []
    for row in readCSV:
        X.append(row)

    del X[0]

print("Test data", X[0])


# The in-data is the acceleration/brake/steering
in_data = []
for row in X:
    in_data.append(row[4:25])

# This is the output data (everything else)
out_data = []
for row in X:
    out_data.append(row[0:3])

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = np.random.random(21)
print(syn0)

for iter in range(1,2):

    # forward propagation
    l0 = np.asarray(in_data[iter])
    print(l0)
    #l1 = nonlin(np.dot(l0,syn0))               TODO: Need to find a way to multiply these!

    # how much did we miss?
    #l1_error = out_data - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    #l1_delta = l1_error * nonlin(l1,True)

    # update weights
    #syn0 += np.dot(l0.T,l1_delta)

#print("Output After Training:")
#print(l1)
