from train import *
from NN import*
import random

import random
import operator
import numpy as np
from NN import *
from train import *
# from neural_try import *

def makepopulation(parents_file = None):
    if parents_file == None:
        layer_info = [[58, 39, 21, 6, 24, 3]]
        iterations = 1000
        lr = 45*10**-7
        populations = {}
        family = []
        for elem in layer_info:
            for i in range(20):
                print(i)
                net = create_nn(iterations, elem, '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/own_races.csv',path_to_filename2 = '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/alpine_own.csv', path_to_filename3 = '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/multiple_Races.csv', lr = lr)
                family.append(net)
    else:
        family = torch.load(parents_file)
    return family

 #[58, 39, 21, 6, 24, 3],


#torch.save(populations[1], 'species_1.pt')
#torch.save(populations[2], 'species_2.pt')
#torch.save(populations[3], 'species_3.pt')
#torch.save(populations[4], 'species_4.pt')
#torch.save(populations[5], 'species_5.pt')

#species_1 = torch.load('species_1.pt')
#print(len(species_1))
#print(len(species_1))
#population=[]
#for key, value in species_1.items():
#    population.append(value)

#torch.save(population, 'species_1.pt')


#
# def makepopulation(generatie, parents_file = None):
#     if parents_file == None:
#         lays = [1,2,3,4,5]
#         nodes = [9,8,7,6,5]
#         populations = {}
#         for j in range(1, 6):
#             pop = []
#             for i in range(20):
#                 layers = []
#                 layers.append(22)
#                 for z in range(j):
#                     layers.append(nodes[z])
#                 layers.append(3)
#                 net = NN(layers)
#                 #create_nn(1000, layers, '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')
#                 create_nn(1000, layers, '/home/student/Documents/new/CI/train_data/aalborg.csv')
#                 pop.append(net)
#             populations[j] = pop
#     else:
#         populations = torch.load(parents_file)
#     return populations


#populations = makepopulation(1)
#for key,val in populations.items():
#     print(key, val)

#print(populations[1][0])

# def selectParents(fitness = None):
#
#     ## INPUT: list of fitness values of networks
#     ## OUTPUT: index of 5 best networks in fitness-list, index of 3 random networks in fitness-list.
#
#     control = fitness.clone() # clone fitness for getting index of random values later on
#
#     index_beste = sorted(range(len(fitness)), key=lambda i: fitness[i])[-5:] # Take 5 best
#
#     for index in sorted(index_beste, reverse=True): # delete them from the fitness set
#         del fitness[index]
#
#     random_selection = np.random.choice(fitness, 3, replace=False) # take 3 random from the remaining set
#     random_selection = list(random_selection)   # make it a list
#     index_random = []
#     for item in random_selection:
#         index_random.append(control.index(item))    # get the index
#
#     return index_beste, index_random




#net = NN([22, 5,3])
#create_nn(1000, [22,5,3], '/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')

#net = mutate(net)






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












def create_child(parent1, parent2, layer_info, num_of_child):
    """
    Create a child model of two parent models.
    Input:
        :param parent1: torch.nn model
        :param patent2: torch.nn model
        :param layer_info: list with information of layers in model
        :param num_of_child: 1 or 2
    """
    # create child model with weights of parent 1
    child = NN(layer_info)
    child.load_state_dict(parent1.state_dict())

    # loop over weights of child
    for k, weight in enumerate(child.parameters()):
        p2_weight = list(parent2.parameters())[k]
        size = weight.data.size()
        # if weight is vector / bias
        if len(size) == 1:
            for i in range(size[0]):
                # set child half of the weight values to weight values of parent 2
                if num_of_child == 1 and i > size[0]/2:
                    weight.data[i] = p2_weight.data[i]
                elif num_of_child == 2 and i <= size[0]/2 :
                    weight.data[i] = p2_weight.data[i]
        # if weight is matrix
        if len(size) == 2:
            for i in range(size[0]):
                for j in range(size[1]):
                    # set child half of the weight values to weight values of parent 2
                    if num_of_child == 1 and i > size[0]/2:
                        weight.data[i,j] = p2_weight.data[i,j]
                    elif num_of_child == 2 and i <= size[0]/2:
                        weight.data[i,j] = p2_weight.data[i,j]
    return child




def mutate(model):
    mutate = random.choice([True, False])
    #print(mutate)
    if mutate:
        for weight in model.parameters():
            size = weight.data.size()

            # if weight is vector / bias
            if len(size) == 1:
                random_select = random.choice(range(round(size[0]/2)))
                random_num1 = random.sample(range(size[0]), random_select)
                random_num2 = random.sample(range(size[0]), random_select)
                print(random_select)
                for i in range(random_select):
                    weight_temp = weight.data[random_num1[i]]
                    weight.data[random_num1[i]] = weight.data[random_num2[i]]
                    weight.data[random_num2[i]] = weight_temp

            # if weight is matrix
            if len(size) == 2:
                indices = [(i,j) for i in range(3) for j in range(3)]
                random_select = random.choice(range(round(len(indices)/2)))
                print(random_select)
                random_tup1 = random.sample(indices, random_select)
                random_tup2 = random.sample(indices, random_select)
                for k in range(random_select):
                    (i1,j1), (i2,j2) = random_tup1[k], random_tup2[k]
                    weight_temp = weight.data[i1,j1]
                    weight.data[i1,j1] = weight.data[i2,j2]
                    weight.data[i2,j2] = weight_temp

    # is returnen nodig aangezien het een object is?
    return model
