import argparse
from src.data_load import load_data
from src.ising import Ising1D

def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="relative path of file data file")
    parser.add_argument("--verbose", help='True/False to turn verbose output on/off')
    return parser.parse_args()

def main(args):
    '''Main function to execute script'''
    input_data = load_data(args)
    verbosity = True if args.verbose != "False" else False

    model = Ising1D(input_data.shape[1], input_data.shape[0])
    model.train_weights(input_data, 250, 0.05, flips_per_site=25, verbose=verbosity)


if __name__ == '__main__':
    args = parse_args()
    main(args)
