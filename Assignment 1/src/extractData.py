'''Handles reading in input data file and hyper-param json file'''
import os
import json
import numpy as np

def get_hyperparameters(args):
    '''Extracts hyperparameters from json file passed in argparse.

    Returns
    ---
    learning_rate: float
        learning rate of the gradient descent
    num_iter: int
        number of iterations for gradient descent
    '''
    json_path = os.path.join(os.getcwd(), str(args.json_path))
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        return json_data['learning rate'], json_data['num iter']
    
def get_input_data(args):
    '''Extracts input data from .in file passed in argparse.

    Returns
    ---
    data_array: numpy array
        array of the extracted data
    '''
    data_path = os.path.join(os.getcwd(), str(args.in_path))
    data_array = np.loadtxt(data_path, delimiter=" ")
    return data_array