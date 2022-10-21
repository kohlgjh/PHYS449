import os
import json
import numpy as np
import torch

def get_hyperparameters(args):
    '''Extracts hyperparameters from json file passed in argparse.

    Returns
    ---
    json_data: dict of json params
    '''
    json_path = os.path.join(os.getcwd(), str(args.json_path))
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        return json_data


def get_data():
    '''Extracts input data from csv

    Returns
    ---
    train_input:
        training data for input
    train_labels:
        labels for the input training data
    test_input:
        testing data to evaluate model
    test_labels:
        labels for the testing data
    '''
    # load csv and split into labels and data
    dataset = np.loadtxt((os.path.join(os.getcwd(), 'inputs\even_mnist.csv')), dtype=np.float32)
    labels = dataset[:,-1]/2
    data = dataset[:,:-1]

    num_data = data.shape[0]
    # perform an 80/20 split on training/test data
    train_data, test_data = data[:int(num_data*0.80), :], data[int(num_data*0.80):, :]
    train_labels, test_labels = labels[:int(num_data*0.80)], labels[int(num_data*0.80):]
    
    # convert to pytorch tensors
    train_input = torch.from_numpy(train_data.astype(np.float32))
    train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
    test_input = torch.from_numpy(test_data.astype(np.float32))
    test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)
    
    return train_input, train_labels, test_input, test_labels
