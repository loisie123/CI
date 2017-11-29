from train import *

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

parent1 = create_nn(1000, [22,6, 6, 8,3], 'train_data/aalborg.csv',path_to_filename2 = 'train_data/alpine-1.csv', path_to_filename3 = 'train_data/f-speedway.csv')
parent2 = create_nn(1000, [22,6, 6, 8,3], 'train_data/aalborg.csv',path_to_filename2 = 'train_data/alpine-1.csv', path_to_filename3 = 'train_data/f-speedway.csv')

layer_info = [22, 6, 6, 8, 3]

child1 = create_child(parent1, parent2, layer_info, 1)
child2 = create_child(parent1, parent2, layer_info, 2)

for weight in child1.parameters():
    print(weight.data)
