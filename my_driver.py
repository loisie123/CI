from pytocl.driver import Driver
from pytocl.car import State, Command
import logging
import torch
import math
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, IntegrationController, DerivativeController
from neural_try import *
from NN import *
from train import *
from ea2 import *
import random


class MyDriver(Driver):

    def __init__(self, logdata=True):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.row = 0
        self.file1 = open('data1.csv', 'a')
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        #self.trainNetwork()
        self.number_of_carstates = 0

        # make a population and choose a model:
        self.populations = makepopulation(1, parents_file ='/home/student/Documents/new/CI/ouwedata.pt')
        #self.populations = makepopulation(1)
        #torch.save(self.population, 'ouwedata.pt')

        #state aanmaken:
        self.begin_damage = 0.1
        self.begin_distance = 0.1
        self.start_carstate = 0.1

        #different kinds of neural networks
        self.species = [1,2,3,4,5]

        self.group = self.species[0]
        self.new_population = {}
        # taka a specie and a indivu of that specie


        self.specie = self.firstmodel(self.populations, self.group)
        self.individu = 0
        self.net = self.specie[self.individu]

        # self.w1 = self.net[0]
        # self.w2 = self.net[1]
        self.model_number = 0
        self.list_of_scores = []

    def firstmodel(self, population, i):
        # takes the population dictionairy and returns species.
        #torch.save(population, 'ouwedata.pt')
        species = population[i]
        print(species)
        return species

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        #make a command.
        command = Command()
        #make input_line
        input_line = [carstate.speed_x,carstate.distance_from_center, carstate.angle]
        for i in range(len(carstate.distances_from_edge)):
            input_line.append(carstate.distances_from_edge[i]   )
    #    for i in range(len(carstate.opponents)):
    #                                                                                               input_line.append(carstate.opponents[i])
        # get output:
        output = self.create_ouput((input_line))

        #make new state

        command.accelerator = output.data[0,0]

        command.brake = output.data[0,1]
        command.steering =  output.data[0,2]
        print("gear ", carstate.gear)
        print("command", command.gear)
        print("accelerator", command.accelerator)
        print("brake", command.brake)
        self.number_of_carstates += 1
        score = self.fitnesfunction(carstate.damage, carstate.distance_raced, self.number_of_carstates, carstate.race_position)
        self.getGear(command.accelerator, carstate.rpm, carstate.gear, command)

        #als de auto stilstaat.

        #and  -50 < carstate.angle < 50
        #wissel van neurale netwerken:

        if self.number_of_carstates == 200  :
            self.on_shotdown(command)
            #self.model_number += 1
            #carstate.damage = 0
            #self.list_of_scores.append(((self.group, self.net), score))
            #self.net = self.changemodel(carstate.damage, carstate.distance_raced ,self.number_of_carstates )
        #v_x = 250

        return command

    def getGear(self, accelerator, rpm, gear, command):
        if gear == 0 and accelerator > 0:
            command.gear = 1
        if gear > 0  and rpm > 8000:
            command.gear += 1
        if accelerator < 0 and gear > 0:
            command.gear -= 1
        elif accelerator < 0 and gear == 0:
            command.gear = -1
        elif accelerator < 0 and gear < 0 :
            command.gear = 1

    def changemodel(self, damage, distance, states):
        """
        This function let the model change.
        """

        self.number_of_carstates = 0
        self.begin_distance = distance- 0.001
        self.start_carstate = states

        if self.individu == (len(self.specie) - 1) and self.group == 5:
            #end is reached:
            print(" alle models have been evaluated")
            #self.EA(self.list_of_scores, self.populations, self.group)
            self.on_shotdown()
        elif self.individu == (len(self.specie) - 1):

            self.individu = 0
            print("next group first individual")
            #self.EA(self.list_of_scores, self.populations, self.group)
            self.list_of_scores = []
            self.group += 1
            return self.populations[self.group][self.individu]
            # all individuals of one group are evaluated.
        else:
            # go to next individual
            self.individu += 1
            print("next indiviudual")
            return self.populations[self.group][self.individu]

    def EA(self, scores, population, group):
        """
        scores: the fitness score of each network
        population: the neural networks of one species.
        """
        print(self.group)

        if self.group == 1:
            layer_info = [22,9,3]
        elif self.group ==2:
            layer_info  =[22,9,8,3]
        elif self.group == 3:
            layer_info   = [22,9,8,7,3]
        elif self.group == 4:
            layer_info = [22,9,8,7,6,3]
        else:
            layer_info = [22,9,8,7,6,5,3]
        print("evaluate networks")
        fitness_scores = []
        netwerk_list = []
        for indx in range(len(scores)):
            fitness_scores.append(scores[indx][1])
            individu = scores[indx][0]
            netwerk_list.append(individu[1])
        index_best, index_worst  = selectParents(fitness_scores)

        # best networks.
        best = []
        for i in range(len(index_best)):
            best.append(netwerk_list[index_best[i]])
        worst = []
        for i in range(len(index_worst)):
            worst.append(mutate(netwerk_list[index_worst[i]]))
        new_pop = []
        parents = best + worst
        print("make children")
        for i in range(8):
            couple = random.sample(parents, 2)
            child1 = create_child(couple[0], couple[1], layer_info, 1)
            new_pop.append(child1)
            child2 = create_child(couple[0], couple[1], layer_info, 2)
            new_pop.append(child2)

        new_pop.append(best[0])
        new_pop.append(best[1])
        new_pop.append(best[2])
        new_pop.append(best[3])
        print("how many children do we have?", len(new_pop))
        self.new_population[group] = new_pop

    def fitnesfunction(self, damage, afstandcenter,carstates, position):
        if self.model_number == 0:
            score = (afstandcenter - damage)
        else:
            score = (afstandcenter - self.begin_distance) - damage /position
        return score

    def create_ouput(self, input_line):
        """
        Function that creates output from an input_line
        """
        tens = torch.FloatTensor(input_line)
        y_variable = torch.autograd.Variable(tens, requires_grad=False)
        ipt = y_variable.view(1, 22)

        out = self.net(ipt)
        # y_pred = ipt.mm(self.w1)
        # out = y_pred.mm(self.w2)
        #output variables 0: acceleration  (has to be zero or 1)
        return out

    def on_restart(self):
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None



    def on_shotdown(self, command):
        """
        functions that is called when the server requested drive shutdown.
        """

        print("ik wil opslaan")
        torch.save(self.new_population, 'children.pt')
        command.meta = 1.0


        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None
