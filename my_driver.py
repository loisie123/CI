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
from ea import *


class MyDriver(Driver):

    def __init__(self, logdata=True):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        #self.trainNetwork()
        self.number_of_carstates = 0

        # make a population and choose a model:
        self.population = makepopulation(generatie = 1, parents_file = '/home/student/CI/lijstvanparent.pt')

        #state aanmaken:
        self.begin_damage = 0.1
        self.begin_distance = 0.1
        self.start_carstate = 0.1

        # maak eerste network
        self.net = self.population[0]
        # self.w1 = self.net[0]
        # self.w2 = self.net[1]
        self.model_number = 0

        self.list_of_scores = []

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

        # get output:
        output = self.create_ouput((input_line))

        #make new state
        command.accelarator = output.data[0,0]
        command.brake = output.data[0,1]
        command.steering =  output.data[0,2]
        self.steer(carstate, 0.0, command)


        #Houdt bij hoeveel carstates er zijn geweest en bereken de score
        self.number_of_carstates += 1
        score = self.fitnesfunction(carstate.damage, carstate.distance_raced, self.number_of_carstates)

        #als de auto stilstaat.
        if 70 < carstate.angle < 120 and carstate.speed_x < 0.0:
            command.gear = -1
        ACC_LATERAL_MAX = 6400 * 5
        v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))

        #wissel van neurale netwerken:
        if self.number_of_carstates > 500 and  -50 < carstate.angle < 50 :
            self.list_of_scores.append(score)
            self.changemodel(carstate.damage, carstate.distance_raced, self.number_of_carstates)
            print("Change the model:")
            print(self.list_of_scores)


            #when last network is reached
            if self.model_number + 1  == len(self.population):
                self.on_shotdown()

        #v_x = 150

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command


    def changemodel(self, damage, distance, states):
        self.model_number += 1
        self.number_of_carstates = 0

        self.begin_damage = damage
        self.begin_distance = distance- 0.001
        self.start_carstate = states
        print("model_number=", self.model_number)
        self.net = self.population[self.model_number]
        # self.w1 = self.net[0]
        # self.w2 = self.net[1]

    def fitnesfunction(self, damage, afstandcenter,carstates):
        if self.model_number == 0:
            score = afstandcenter - damage
        else:
            score = (afstandcenter - self.begin_distance) - (damage- self.begin_damage)
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

    def on_shotdown(self):
        """
        functions that is called when the server requested drive shutdown.
        """

        print("ik wil opslaan")
        torch.save(self.population, 'lijstvanparent.pt')
        index_best, index_worst  = selectParents(self.list_of_scores)
        print("index_best", index_best)
        print("index_worst", index_worst)
        best = []
        #find out which networks are the best and worst
        for i in range(len(index_best)):
            best.append(self.population[index_best[i]])
        worst = []
        for i in range(len(index_worst)):
            worst.append(self.population[index_worst[i]])

        #mutate networks
        for network in worst:
            mutate(network.paramaters)

        parents = best + worst

        #randomly select parents
        #chidl1, child2 = breead(parent1, parent2)
        #child.appen(child1)
        #child.append(child2)
        #torch.save(parents, 'lijstvanparent.pt')
        # save children.


        print("ik wil weten wanneer ik aangeroepen word.s")

    # def drive(self, carstate: State) -> Command:
    #     """
    #     Produces driving command in response to newly received car state.
    #
    #     This is a dummy driving routine, very dumb and not really considering a
    #     lot of inputs. But it will get the car (if not disturbed by other
    #     drivers) successfully driven along the race track.
    #     """
    #     command = Command()
    #
    #     command = Command()
    #
    #
    #
    #     self.steer(carstate, 0.0, command)
    #     print("print hij iets")
    #     ACC_LATERAL_MAX = 6400 * 5
    #     v_x = min(200, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
    #     v_x = 200
    #
    #     self.accelerate(carstate, v_x, command)
    #
    #     if self.data_logger:
    #         self.data_logger.log(carstate, command)
    #
    #     #get a line to print the right statements to the carstate.
    #     input_line = [self.accelerate(carstate, v_x, command),0 , self.steer(carstate, 0.0, command),carstate.speed_x,carstate.distance_from_center, carstate.angle]
    #     for i in range(len(carstate.distances_from_edge)):
    #         input_line.append(carstate.distances_from_edge[i])
    #     for i in range(len(carstate.opponents)):
    #         input_line.append(carstate.opponents[i])
    #
    #
    #     return command
    #
    # def write_file(self, input_line):
    #     files = open("testfile.txt","w")
    #     files.write(input_line, "\n")
    #     files.close()
    #
    #     #function that writes info in file
