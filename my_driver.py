from pytocl.driver import Driver
from pytocl.car import State, Command
import logging
import math
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, IntegrationController, DerivativeController
from neural_try import *

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

        self.w1, self.w2 = NN('/home/student/Documents/CI/torcs-server/torcs-client/train_data/aalborg.csv')


    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        input_line = [carstate.speed_x,carstate.distance_from_center, carstate.angle]
        for i in range(len(carstate.distances_from_edge)):
            input_line.append(carstate.distances_from_edge[i]   )

        output = self.create_ouput((input_line))
        print(output)

        print("eerste command",  command)
        accelarator = output.data[0,0]
        breake = output.data[0,1]
        if accelarator > 1:
            command.accelerator = 1.0
            print(command.accelerator)
        elif accelarator < 0.0:
            command.accelerator = 0.0
        else:
            command.accelarator = accelarator
        print("accelerator", output.data[0,0], command.accelerator)

        if breake > 1.0:
             command.brake = 1.0
        elif breake < 0.0:
            command.brake = 0.0
        else:
            command.brake = breake
        print("brake:" , output.data[0,1], command.brake)

        command.steering =  output.data[0,2]
        print("steer:" , output.data[0,2], command.steering)

        print ("tweede command:", command)


        self.steer(carstate, 0.0, command)

	# ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    def create_ouput(self, input_line):
        """
        Function that creates output from an input_line
        """
        tens = torch.FloatTensor(input_line)
        y_variable = torch.autograd.Variable(tens, requires_grad=False)
        ipt = y_variable.view(1, 21)
        y_pred = ipt.mm(self.w1)
        out = y_pred.mm(self.w2)
        #output variables 0: acceleration  (has to be zero or 1)


        return out
