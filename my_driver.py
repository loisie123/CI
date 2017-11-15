# from pytocl.driver import Driver
# from pytocl.car import State, Command
from neural_try import *
import csv
import numpy

# class MyDriver(Driver):
#     """
#     """
#
#     def __init__(self, logdata =True):
#         self.steering_ctrl = CompositeController(
#             ProportionalController(0.4),
#             IntegrationController(0.2, integral_limit=1.5),
#             DerivativeController(2)
#         )
#         self.acceleration_ctrl = CompositeController(
#             ProportionalController(3.7),
#         )
#         self.data_logger = DataLogWriter() if logdata else None
#
#
#     def drive(self, carstate: State):
#
#         """
#         Produces driving command in response to newly received car state.
#
#         This is a dummy driving routine, very dumb and not really considering a
#         lot of inputs. But it will get the car (if not disturbed by other
#         drivers) successfully driven along the race track.
#         """
#
#         command = Command()
#
#         self.steer(carstate, 0.0, command)
#
#         # ACC_LATERAL_MAX = 6400 * 5
#         # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
#         v_x = 80
#
#         self.accelerate(carstate, v_x, command)
#
#         if self.data_logger:
#             self.data_logger.log(carstate, command)
#
#         return command
#
#
#     def train_NN(path_to_filename):
#         w1, w2 = NN(path_to_filename)
#         return w1, w2
#
#
#     def create_ouput(input_line):
#         """
#         Function that creates output from an input_line
#         """
#         w1, w2 = train_NN('train_data/aalborg.csv')
#         y_variable = torch.autograd.Variable(input_line, requires_grad=False)
#         ipt = y_variable.view(1, 21)
#         y_pred = ipt.mm(w1)
#         out = y_pred.mm(w2)
#         #output variables 0: acceleration  (has to be zero or 1)
#
#
#         return out
#




with open('train_data/aalborg.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    X = []
    for row in readCSV:
        X.append(row)

    del X[0]

    # The in-data are all the external variables
in_data = []
for row in X:
    in_data.append(row[4:25])
firstline = torch.FloatTensor(in_data[0])

# function that calls on the neural network

def train_NN(path_to_filename):
    w1, w2 = NN(path_to_filename)
    return w1, w2

def create_ouput(input_line):
     """
     Function that creates output from an input_line
     """
     w1, w2 = train_NN('train_data/aalborg.csv')
     y_variable = torch.autograd.Variable(input_line, requires_grad=False)
     ipt = y_variable.view(1, 21)
     y_pred = ipt.mm(w1)
     out = y_pred.mm(w2)
     #output variables 0: acceleration  (has to be zero or 1
     return out



output = create_ouput(firstline)
accelarator = output.data[0,0]

if accelarator > 0.5:
    print(True)









#input new data into the model and get output that calls on the accelaration methods.
