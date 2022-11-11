'''Sublass structure for the neural network model'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(2, 128, num_layers=2, batch_first=True) # input size is either 2 or 1?
        self.rnn_to_fcl = nn.Linear(128, 32) # hidden layer to 32 
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, input_size):
        outputs, (h_n,c_n)= self.lstm(x)
        output = self.rnn_to_fcl(h_n[-1])
        output = torch.reshape(output, (input_size, 16, 2)) # reshape 32 into 16x2 one-hots
        return self.softmax(output)

    def reset(self):
        self.lstm.reset_parameters()
        self.rnn_to_fcl.reset_parameters()