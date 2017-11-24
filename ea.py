import random
import operator
import numpy as np

def makepopulation(first = False):
    #amaakt de eerste populatie
    return population

def fitness():
    return fitness

def selectParents(fitness = None):

    ## Returns indexes in fitness list of selected parents (5 best and 3 random) ##

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

def mutate(random):
    #mutate de random selection

    return network


def breed(network1, network2):
    return child1, child2



def main():
    #dit is de main function.
    return new_popultion
