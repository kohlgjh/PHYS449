
import argparse
import os
import torch

def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser(usage='main.py [-h] [--param param.json] [--train_size INT] [--test_size INT] [--seed INT]')
    parser.add_argument("--param", help="relative path of file containing hyperparameters")
    parser.add_argument("--train_size", type=int, help="size of the generated training set")
    parser.add_argument("--test_size", type=int, help="size of the generated test set")
    parser.add_argument("--seed", type=int, help="random seed used for creating the datasets")
    return parser.parse_args()

def main(args):
    '''Main entry point to assignment 3 code'''

if __name__ == '__main__':
    args = parse_args()
    main(args)
