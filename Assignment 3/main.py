
import argparse
import os
import torch
from src.model import Net
from src.extract_parameters import get_parameters
from src.data_processing import generate_data_loader
from src.train_and_test import train_and_test
from src.plot import generate_loss_report

def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser(usage='main.py [-h] [--param param.json] [--train_size INT] [--test_size INT] [--seed INT]')
    parser.add_argument("--param", help="relative path of file containing hyperparameters")
    parser.add_argument("--train_size", type=int, help="size of the generated training set")
    parser.add_argument("--test_size", type=int, help="size of the generated test set")
    parser.add_argument("--seed", type=int, help="random seed used for creating the datasets")
    parser.add_argument("--device", help="cuda or cpu - device to run on")
    return parser.parse_args()

def main(args):
    '''Main entry point to assignment 3 code'''
    model = Net().to(args.device)
    params = get_parameters(args)

    # data generation
    train_loader, test_loader = generate_data_loader(args.train_size, args.test_size, args.seed, args.device, int(params['batch_size']))

    # train and test
    obj_vals, cross_vals, accuracy = train_and_test(model, train_loader, test_loader, params)

    # making plots
    generate_loss_report(obj_vals, cross_vals, accuracy)


if __name__ == '__main__':
    args = parse_args()
    main(args)
