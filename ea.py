import random
import operator
import numpy as np
from NN import *
from train import *
# from neural_try import *


def makepopulation(generatie, parents_file = None):
    if parents_file == None:
        pop = []
        for j in range(1, 5):
            array = np.zeros(2+j)
            array[0] = 22
            array[-1] = 3
            print(j)
            for i in range(20):
                if j == 1:
                    array[j] = 5
                if j == 2:
                    array[j-1] = 6
                    array[j] = 5
                if j == 3:
                    array[j-2] = 7
                    array[j-1] = 6
                    array[j] = 5
                if j == 4:
                    array[j-3] = 8
                    array[j-2] = 7
                    array[j-1] = 6
                    array[j] = 5
                if j == 5:
                    array[j-4] = 9
                    array[j-3] = 8
                    array[j-2] = 7
                    array[j-1] = 6
                    array[j] = 5

                #w1,w2 = NN( ,path_to_filename2 = '/home/student/CI/train_data/alpine-1.csv', path_to_filename3 = '/home/student/CI/train_data/f-speedway.csv')
                #net = (w1, w2)


                net = NN(array)
                create_nn(1000, array , '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/aalborg.csv',path_to_filename2 = '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/alpine-1.csv', path_to_filename3 = '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/f-speedway.csv')

                #main1(1000, 5, '/home/koen/Documents/ComputationalIntelligence/CI/train_data/aalborg.csv', path_to_filename2= '/home/koen/Documents/ComputationalIntelligence/CI/train_data/alpine-1.csv', path_to_filename3 = '/home/koen/Documents/ComputationalIntelligence/CI/train_data/f-speedway.csv' )

                #make a network
                #net = Net(forward_info)

                #pop.append(net)
    else:
        pop = torch.load(parents_file)
    return pop


def selectParents(fitness = None):

    ## INPUT: list of fitness values of networks
    ## OUTPUT: index of 5 best networks in fitness-list, index of 3 random networks in fitness-list.

    control = fitness[:] # clone fitness for getting index of random values later on

    index_beste = sorted(range(len(fitness)), key=lambda i: fitness[i])[-5:] # Take 5 best

    for index in sorted(index_beste, reverse=True): # delete them from the fitness set
        del fitness[index]

    random_selection = np.random.choice(fitness, 3, replace=False) # take 3 random from the remaining set
    random_selection = list(random_selection)   # make it a list
    index_random = []
    for item in random_selection:
        index_random.append(control.index(item))    # get the index

    return index_beste, index_random

def mutate(net, first = False):

    ## INPUT: list of weights arrays of network, can be any number.
    ## OUTPUT: list of weights arrays (mutated with probability .2) of mutated network


    if first == True:
        para = list(net.parameters()) # Unpack the parameters
    else:
        para = net # is already a normal list
    params = []
    mutation_indicator = np.random.choice(2, 1, p=[0, 1.0]) # mutation with probability of .2


    print(list(net.parameters()))
    if mutation_indicator == 1: # if mutate
        for idx, mat in enumerate(list(net.parameters())):
            mat = mat.data.cpu().numpy()
            mat = np.random.permutation(mat)
            params.append(torch.nn.Parameter(torch.from_numpy(mat)))


            #net.register_parameter(params)

        #list(net.parameters()) = para
        #net = para

        return net # return list of mutated weight matrices

    else: # if not mutate
        net = para
        return net # return original network


net = NN([22, 5,3])
create_nn(1000, [22,5,3], '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')

net = mutate(net)




def breed(network1, network2):

    ## INPUT: list of weights arrays of parent network 1 and list of weights array of parent network 2.
    ## OUTPUT: list of weights arrays of child network 1 and list of weights arrays of child network 2

    # Child 1
    CH1 = []    # list to store child 1's weight matrices
    count = 0
    total = 0
    for ind, mat in enumerate(network1):    # for every weight matrix
        mat = mat.data.numpy()
        for idx, row in enumerate(mat): # for every row in current weight matrix
            selection_indicator = np.random.choice(2, 1, p=[.5, .5]) # choose parent (0 for parent 1, 1 for parent 2)
            if selection_indicator == 0:
                count += 1
                total += 1
                mat[idx] = network1[ind].data.numpy()[idx] # take row from parent 1
            else:
                total += 1
                mat[idx] = network2[ind].data.numpy()[idx] # take row from parent 2
        child1 = torch.nn.Parameter(torch.from_numpy(mat))
        CH1.append(child1)

    print("Child 1: ",count/total * 100,"% from parent 1. ", 100 - (count/total * 100), '% from parent 2.')

    # Child 2, same principle
    CH2 = []
    count = 0
    total = 0
    for ind, mat in enumerate(network1):
        mat = mat.data.numpy()
        for idx, row in enumerate(mat):
            selection_indicator = np.random.choice(2, 1, p=[.5, .5]) # 0 for parent 1, 1 for parent 2
            if selection_indicator == 0:
                count += 1
                total += 1
                mat[idx] = network1[ind].data.numpy()[idx] # take row from parent 1
            else:
                total += 1
                mat[idx] = network2[ind].data.numpy()[idx] # take row from parent 2
        child2 = torch.nn.Parameter(torch.from_numpy(mat))
        CH2.append(child2)

    print("Child 2: ",count/total * 100,"% from parent 1. ", 100 - (count/total * 100), '% from parent 2.')

    return CH1, CH2

def selectSurvivors(fitness = None):

    ## INPUT: list of fitness values of networks
    ## OUTPUT: index of 4 best networks in fitness-list (survivors)

    index_beste = sorted(range(len(fitness)), key=lambda i: fitness[i])[-4:] # Take 4 best

    return index_beste

# TODO: Run for example:
#
#
#net = Net()

# net = NN([22, 5,3])
# create_nn(1000, [22,5,3], '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/aalborg.csv',path_to_filename2 = '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/alpine-1.csv', path_to_filename3 = '/home/student/Documents/CI/CI/torcs-server/torcs-client/train_data/f-speedway.csv')
#
# params1 = list(net.parameters())
# params2 = mutate(net, first = True)
# params3 = mutate(params2)

## Mutation demonstration

def mutation_demonstration():

    x = mutate(params3)
    for i in range(0,1000):
        x = mutate(x)
        if i == 0:
            print("Begin matrix:")
            print(x)
        if i == 999:
            print("End matrix:")
            print(x)
    return

#mutation_demonstration()

#c1, c2 = breed(params1, params3)
#pop = makepopulation(1)
#print(len(pop))
