# from pytocl.driver import Driver
# from pytocl.car import State, Command
from neural_try import *


#
# class MyDriver(Driver):
#     print("hello")
#     # Override the `drive` method to create your own driver
#
#     # def drive(self, carstate: State) -> Command:
#     #     # Interesting stuff
#     #     command = Command(...)
#     #     return command







with open('/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv') as csvfile:
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
    w1, w2 = train_NN('/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')
    y_variable = torch.autograd.Variable(input_line, requires_grad=False)
    ipt = y_variable.view(1, 21)
    y_pred = ipt.mm(w1)
    out = y_pred.mm(w2)
    #output variables 0: acceleration  (has to be zero or 1)
    print(out[0,0])
    if out[0,0] > 0.5:
        out[0,0] = 1
    else:
        out[0,0] = 0
    print(out[0,0])

    return out







create_ouput(firstline)

# input new data into the model and get output that calls on the accelaration methods.
