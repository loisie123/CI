# from pytocl.driver import Driver
# from pytocl.car import State, Command
from neural_try import *



#class MyDriver(Driver):
#    print("hello")
    # Override the `drive` method to create your own driver

    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command




# function that calls on the neural network
def train_NN(path_to_filename):
    w1, w2 = NN(path_to_filename)
    print(w1)
    print(w2)



NN = train_NN('/Users/loisvanvliet/Documents/studie/2017:2018/Computational intelligence/CI/train_data/aalborg.csv')


# We have to start the car.



# input new data into the model and get output that calls on the accelaration methods.
