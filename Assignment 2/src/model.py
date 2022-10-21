'''Class for the neural net model'''
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # defining our layers
        self.fc1 = nn.Linear(196, 784) # Go from input size to 4 * input size 
        self.fc2 = nn.Linear(784, 40) # Compress down to just 40 neurons
        self.fc3 = nn.Linear(40, 5) # final compress from 40 to 5 output options

    def forward(self, x): # x is input vector
        h1 = func.relu(self.fc1(x)) # relu for layer 1
        h2 = func.relu(self.fc2(h1))
        y = func.log_softmax(self.fc3(h2), dim=1) # final evaluation done with softmax
        return y

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()