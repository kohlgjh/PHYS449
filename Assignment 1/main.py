import argparse
import os
import json
import numpy as np

def parse_args():
    '''Parses command line args for input path and json path'''
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="the relative input file path")
    parser.add_argument("json_path", help="the relative json file path")
    args = parser.parse_args()
    return args

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

def main(args):
    '''Main entry point to assignment 1 code'''
    
    # extract data and hyperparameters from passed files
    learning_rate, num_iter = get_hyperparameters(args)
    data_array = get_input_data(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
