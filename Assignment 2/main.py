import argparse
import os
import torch

from src.extractData import get_hyperparameters, get_data
from src.model import Net
from src.train_and_test import train_and_test
from src.report import generate_report

def parse_args():
    '''Parses command line args for json path'''
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="the relative json file path")
    args = parser.parse_args()
    return args

def main(args):
    '''Main entry point to assignment 2 code'''
    # extracting data and parameter dictionary
    train_input, train_labels, test_input, test_labels = get_data()
    params = get_hyperparameters(args)

    # setting up model and hyperparamters
    model = Net().to(torch.device("cpu"))
    num_epochs = int(params['num_epochs'])
    display_epochs = int(params['display_epochs'])
    verbose = bool(params['verbose'])
    learning_rate = float(params['learning_rate'])

    # undergo training and testing
    obj_vals, cross_vals, accuracy = train_and_test(model, train_input, train_labels, test_input, test_labels,
                                          num_epochs, display_epochs, verbose, learning_rate)

    # produce the printout of the training results
    generate_report(obj_vals, cross_vals, accuracy, os.getcwd(), num_epochs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
