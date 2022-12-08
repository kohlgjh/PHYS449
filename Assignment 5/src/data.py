import os
import numpy as np
import torch

def load_data(device, batch_size=200):
    '''Loads data from csv and returns torch data loader and original data'''
    raw_data = np.loadtxt(os.path.join(os.getcwd(), r'data\even_mnist.csv'), dtype=str)

    # separate label from images
    input = raw_data[:, :-1].astype(float)/255 # normalize
    label = raw_data[:, -1:].astype(float)

    # reshape the flattened images into 2D
    input = input.reshape((len(label), 1, 14, 14))

    # load into torch tensors
    input = torch.from_numpy(input).type(torch.float).to('cuda')
    label = torch.from_numpy(label).type(torch.float).to('cuda')

    # setup data loader
    input_set = torch.utils.data.TensorDataset(input, label)
    input_loader = torch.utils.data.DataLoader(input_set, batch_size=batch_size, shuffle=True)

    return input_loader, input