import random
import operator
import numpy as np

def makepopulation(first = False):
    #amaakt de eerste populatie
    return population

def fitness():
    return fitness

def selectParents(fitness = None):

    ## INPUT: list of fitness values of networks
    ## OUTPUT: index of 5 best networks in fitness-list, index of 3 random networks in fitness-list.

    control = fitness[:]

    index_beste = sorted(range(len(fitness)), key=lambda i: fitness[i])[-5:]

    for index in sorted(index_beste, reverse=True):
        del fitness[index]

    random_selection = np.random.choice(fitness, 3, replace=False)
    random_selection = list(random_selection)
    index_random = []
    for item in random_selection:
        index_random.append(control.index(item))

    return index_beste, index_random

def mutate(weights_matrices):

    ## INPUT: list of weights arrays of network, can be any number.
    ## OUTPUT: list of weights arrays (mutated with probability .2) of mutated network.

    mutation = []

    mutation_indicator = np.random.choice(2, 1, p=[.8, .2])

    if mutation_indicator == 1:
        print("MUTATION")
        for idx, mat in enumerate(weights_matrices):
            mat = np.asarray(mat)
            new = np.random.permutation(mat)
            mutation.append(new)

        return mutation

    else:
        return weights_matrices

def breed(network1, network2): # TODO: Needs to be finished

    ## INPUT: list of weights arrays of parent network 1 and list of weights array of parent network 2.
    ## OUTPUT: list of weights arrays of child network 1 and list of weights arrays of child network 2

    # Child 1
    CH1 = []
    for ind, mat in enumerate(arr1):
        child1 = np.zeros((mat.shape[0], mat.shape[1]))
        for idx, row in enumerate(mat):
            selection_indicator = np.random.choice(2, 1, p=[.8, .2]) # 0 for parent 1, 1 for parent 2
            if selection_indicator == 0:
                child1[idx] = arr1[ind][idx]
            else:
                child1[idx] = arr2[ind][idx]
        CH1.append(child1)

    # Child 2
    CH2 = []
    for ind, mat in enumerate(arr1):
        child2 = np.zeros((mat.shape[0], mat.shape[1]))
        for idx, row in enumerate(mat):
            selection_indicator = np.random.choice(2, 1, p=[.8, .2]) # 0 for parent 1, 1 for parent 2
            if selection_indicator == 0:
                child2[idx] = arr1[ind][idx]
            else:
                child2[idx] = arr2[ind][idx]
        CH2.append(child2)

    return CH1, CH2

def main():
    #dit is de main function.
    return new_popultion
