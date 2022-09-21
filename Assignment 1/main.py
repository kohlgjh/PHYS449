import argparse
import numpy as np
from src.extractData import get_hyperparameters, get_input_data

def parse_args():
    '''Parses command line args for input path and json path'''
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="the relative input file path")
    parser.add_argument("json_path", help="the relative json file path")
    args = parser.parse_args()
    return args


def main(args):
    '''Main entry point to assignment 1 code'''
    
    # extract data and hyperparameters from passed files
    learning_rate, num_iter = get_hyperparameters(args)
    data_array = get_input_data(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
