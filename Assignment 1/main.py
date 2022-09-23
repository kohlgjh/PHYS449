import argparse
import numpy as np
import os

from src.extractData import get_hyperparameters, get_input_data
from src.linearRegression import LinearRegressionModel
from src.gradientDescent import GradientDescentModel
from src.exportData import export

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

    # the two models
    lgModel = LinearRegressionModel(data_array)
    gdModel = GradientDescentModel(data_array, learning_rate, num_iter)

    # export results
    out_path = os.path.join(os.getcwd(), str(args.json_path).split('/')[-1].split('.')[0] + '.out')
    export(out_path, lgModel.w_star, gdModel.w_star)



if __name__ == '__main__':
    args = parse_args()
    main(args)
