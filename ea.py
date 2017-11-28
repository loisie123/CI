import random
import operator
import numpy as np
from NN import *
from train import *
# from neural_try import *


def makepopulation(generatie, parents_file = None):
    if parents_file == None:
        pop = []
        for i in range(10):
            #w1,w2 = NN( ,path_to_filename2 = '/home/student/CI/train_data/alpine-1.csv', path_to_filename3 = '/home/student/CI/train_data/f-speedway.csv')
            #net = (w1, w2)

            # this must be with Mirthes network

            net = Net()
            main1(1000, 5, '/home/student/CI/train_data/aalborg.csv', path_to_filename2= '/home/student/CI/torcs-server/torcs-client/train_data/alpine-1.csv', path_to_filename3 = '/home/student/CI/train_data/f-speedway.csv' )

            #make a network
            #net = Net(forward_info)

            pop.append(net)
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



def mutate(net):

    ## INPUT: list of weights arrays of network, can be any number.
    ## OUTPUT: list of weights arrays (mutated with probability .2) of mutated network.

    mutation = []

    mutation_indicator = np.random.choice(2, 1, p=[0, 1]) # mutation with probability of .2
    para = list(net.parameters())

    if mutation_indicator == 1: # if mutate
        mutation = []
        for idx, mat in enumerate(list(net.parameters())):
            mat = mat.data.numpy()
            np.random.permutation(mat)

            net.parameters()[idx] = torch.from_numpy(mat)
            #print("Parameters:" , list(net.parameters())[0])


        net.parameters = para

        #print ("helloooo", para)    # shuffle array
            #  # add shuffled matrix to list
        return net # return list of mutated weight matrices

    else: # if not mutate
        return net # return original network

def breed(network1, network2):

    ## INPUT: list of weights arrays of parent network 1 and list of weights array of parent network 2.
    ## OUTPUT: list of weights arrays of child network 1 and list of weights arrays of child network 2

    # Child 1
    CH1 = []    # list to store child 1's weight matrices
    for ind, mat in enumerate(list(net.parameters())):    # for every weight matrix
        child1 = np.zeros((mat.shape[0], mat.shape[1])) # create zero matrix
        for idx, row in enumerate(mat): # for every row in current weight matrix
            selection_indicator = np.random.choice(2, 1, p=[.5, .5]) # choose parent (0 for parent 1, 1 for parent 2)
            if selection_indicator == 0:
                child1[idx] = network1[ind][idx] # take row from parent 1
            else:
                child1[idx] = network2[ind][idx] # take row from parent 2
        CH1.append(child1)

    # Child 2, same principle
    CH2 = []
    for ind, mat in enumerate(network1):
        child2 = np.zeros((mat.shape[0], mat.shape[1]))
        for idx, row in enumerate(mat):
            selection_indicator = np.random.choice(2, 1, p=[.5, .5]) # 0 for parent 1, 1 for parent 2
            if selection_indicator == 0:
                child2[idx] = network1[ind][idx]
            else:
                child2[idx] = network2[ind][idx]
        CH2.append(child2)

    return CH1, CH2

def selectSurvivors(fitness = None):

    ## INPUT: list of fitness values of networks
    ## OUTPUT: index of 4 best networks in fitness-list (survivors)

    index_beste = sorted(range(len(fitness)), key=lambda i: fitness[i])[-4:] # Take 4 best

    return index_beste



net = Net()

main1(1000, 5, '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')
params1 = list(net.parameters())



mutate(net)

params2 = list(net.parameters())
