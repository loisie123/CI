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

    ## INPUT: list of weights matrices of network, can be any number.
    ## OUTPUT: list of weights matrices (mutated with probability .2) of mutated network.

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

def breed(network1, network2):
    return child1, child2



def main():
    #dit is de main function.
    return new_popultion
