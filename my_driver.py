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
import os
from pathlib import Path
from presButton import *



class MyDriver(Driver):

    def __init__(self, logdata=True):

        self.data_logger = DataLogWriter() if logdata else None


        # comment on of the two options below
        #first time:
        #self.populations = makepopulation()
        #torch.save(self.populations, 'species_extra_2.pt')
        # torch.save(self.populations[1], 'species_1.pt')
        # torch.save(self.populations[2], 'species_2.pt')
        # torch.save(self.populations[3], 'species_3.pt')
        # torch.save(self.populations[4], 'species_4.pt')
        # torch.save(self.populations[5], 'species_5.pt')


        #group 1.
        #
        self.layer_info = [58, 27, 36, 24, 3]
        self.population = torch.load('/home/student/Desktop/CI/complete_2_family.pt')
        self.filename = '/home/student/Desktop/CI/complete_2_family.pt'
        self.file1 ='fitnes_1_compleet.csv'

        # #group 2:

        # self.layer_info  =[22,9,8,3]
        # self.population = torch.load('/home/student/Documents/new/CI/species_2.pt')
        # self.filename = '/home/student/Documents/new/CI/species_2.pt'

        # #group 3:
        #self.layer_info  = [58,33,9,15,30, 6, 3]
        #self.population = torch.load('/home/student/Desktop/CI/species_3.pt')
        #self.filename = '/home/student/Desktop/CI/species_3.pt'
        # self.file1 ='fitneswerkthet.csv'

        # #group 4
        # self.layer_info = [22,9,8,7,6,3]
        # self.population = torch.load('/home/student/Documents/new/CI/species_4.pt')
        # self.filename = '/home/student/Documents/new/CI/species_4.pt'
        # self.file1 = open('fitnes_4.csv', 'a')

        # #group 5:
        # self.layer_info = [22,9,8,7,6,5,3]
        # self.population = torch.load('/home/student/Documents/new/CI/species_5.pt')
        # self.filename = '/home/student/Documents/new/CI/species_5.pt'

        # when all these files exists:
        self.net, self.population = self.getModel(self.population, self.filename)

        #2. when there already a population file :
        #self.populations = makepopulation(1, parents_file ='/home/student/Documents/new/CI/ouwedata.pt')

        self.number_of_carstates = 0


    def getModel(self, population, filename):
        """
        returns the selected netwerk for this drive and saves the new population
        population: one of the species (1,2,3,4,5 hidden layers)

        """

        print("getting the right network and popping the network")
        print (len(population))
        net = population[0]
        population.pop(0)
        torch.save(population, filename)
        return net, population


    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        #print(carstate.current_lap_time)
        #make a command.
        command = Command()

        #make input_line.
        input_line = [carstate.speed_x,carstate.distance_from_center, carstate.angle]
        for i in range(len(carstate.distances_from_edge)):
            input_line.append(carstate.distances_from_edge[i])
        for i in range(len(carstate.opponents)):
            input_line.append(carstate.opponents[i])

        # get output:
        output = self.create_ouput((input_line))

        #make new state
        command.accelerator = output.data[0,0]
        #print("accelerator", command.accelerator)
        command.brake = output.data[0,1]
        #print("brake", command.brake)
        command.steering =  output.data[0,2]
        #print(command.steering)
        self.getGear(carstate, command)

        #calculate score
        self.number_of_carstates += 1
        score = self.fitnesfunction(carstate.damage, carstate.distance_raced, carstate.race_position)
        #score = self.fitnesfunction(carstate.distance_raced, self.number_of_carstates, carstate.race_position)

        # when a certain level of damage is done or when the car is not moving:
        if self.number_of_carstates == 2000:

            #save changes:
            self.saveModel(self.net, score)

            # checks if last individu is examined:
            if self.emptyList(self.population) == True:
                print("lijst is leeeg")
                parents = torch.load('parents_file_1.pt')
                print("parents:", parents)
                new_population = self.Evolutionair(parents, self.layer_info)

                self.saveNew(new_population, self.filename)
                print("next generation is comming")
                self.on_shotdown(command)
                # save it in the right file.

            else: #when there are still neural networks available
                self.on_shotdown(command)


        return command

    def getGear(self, carstate, command):
        """
        function that calculates which gear is neccesary.

        carstate: the current carstate:
        command: the new command.
        """

        #print("carstate rpm:", carstate.rpm)
        if command.accelerator > 0 and carstate.rpm > 8000:
            command.gear = carstate.gear + 1
            #print("ga eens naar een hogere versnelling")
        elif command.accelerator < 0 and carstate.rpm < 2500:
            command.gear = carstate.gear - 1
            #print("hij gaat weer door")
        else:
            command.gear = carstate.gear

    def saveNew(self, children, filename):
        #save the children in two, files
        torch.save(children, filename)
        torch.save(children, 'last_generation.pt')
        os.remove('parents_file_1.pt')



    def saveModel(self, net, fitnes):
        """
        this function saves the current neural network

        net: the current networ
        fitnes: fitness score
        """

        my_file = Path('parents_file_1.pt')
        if my_file.is_file():  # this means a file exists
            old_net = torch.load('parents_file_1.pt')
            new_net = (net,fitnes)
            old_net.append(new_net)
            torch.save(old_net, 'parents_file_1.pt')
        else:
            # this is the first model that is analysed.
            new_net = [(net,fitnes)]
            torch.save(new_net, 'parents_file_1.pt' )

    def emptyList(self, population):
        """
        function that checks if all the individuls of a certain population are examined.
        population: list of individuals
        """
        print(len(population))
        if len(population) == 0:
            return True
        else:
            return False

    def makefile(self, output):
        my_file = Path(self.file1)
        if my_file.is_file():
            print("hij bestaat")
            csvfile = open(self.file1, 'a')
            writer = csv.writer(csvfile)
            writer.writerow(output)
            csvfile.close()
        else:
            print("hij bstaat niet")
            csvfile = open(self.file1, 'w')
            writer = csv.writer(csvfile)
            writer.writerow(output)
            csvfile.close()


    def Evolutionair(self, parents, layers):
        """
        function that returns children.

        population: the neural networks of one species.
        layers: layer info that the current neural network has.
        """
        #make fitnes listst and parentslists
        list_fitnes = []
        list_parents = []
        for net_parents in parents:
            list_parents.append(net_parents[0])
            list_fitnes.append(net_parents[1])

        print("fitnes functions", list_fitnes)
        self.makefile(list_fitnes)

        index_best, index_worst  = selectParents(list_fitnes)

        # best networks.
        best = []
        for i in range(len(index_best)):
            best.append(list_parents[index_best[i]])
        worst = []

        #worst networks
        for i in range(len(index_worst)):
            worst.append(mutate(list_parents[index_worst[i]]))

        #new population is a list.
        new_pop = []
        parents = best + worst

        print("make children ")

        # it takes 2 parents (randomly) and makes to children of them
        # so we have 16 children
        coupleset = []
        i = 0
        while i != 16:
            couple = random.sample(parents, 2)
            if couple not in coupleset:
                couple = random.sample(parents, 2)
                child1 = create_child(couple[0], couple[1], layers, 1)
                new_pop.append(child1)
                child2 = create_child(couple[0], couple[1], layers, 2)
                new_pop.append(child2)
                coupleset.append(couple)
                print("couple bestaat nog niet")
                i = i +1
                print(i)
            else:
                i = i
                print("couple bestaat wel")

        # for i in range(16):
        #     child1 = create_child(couple[0], couple[1], layers, 1)
        #     new_pop.append(child1)
        #     child2 = create_child(couple[0], couple[1], layers, 2)
        #     new_pop.append(child2)

        # we append the four best neural networks.
        new_pop.append(best[0])
        new_pop.append(best[1])
        new_pop.append(best[2])
        new_pop.append(best[3])
        new_pop.append(best[4])
        new_pop.append(best[5])
        new_pop.append(best[6])
        new_pop.append(best[7])
        print ("new", len(new_pop))
        return new_pop


    def fitnesfunction(self, damage, afstandcenter, position):
        """
        function that calculates the score of the network.
        damage: the damage that is done on the car.
        afstandcenter:  the distance the car has raced.
        position: the position of the car.
        """

        score = (afstandcenter - (damage/500))/position
        if score == 0:
            score = 0
        return score

    def create_ouput(self, input_line):
        """
        Function that creates output from an input_line
        inputline: consists of speed_x, carstate_distance_from_center, all distances to edges
                and all distances to opponents.
        """
        tens = torch.FloatTensor(input_line)
        y_variable = torch.autograd.Variable(tens, requires_grad=False)
        ipt = y_variable.view(1, 58)

        out = self.net(ipt)
        return out

    def on_restart(self):
        """
        function that makes the car restart the race.
        """

        os.system("./torcs-server/torcs-client/start.sh")
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None



    def on_shotdown(self, command):
        """
        functions that is called when the server requested drive shutdown.
        """
        command.meta = 1.0
