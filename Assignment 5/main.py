import argparse
import os
from src.model import VAE
from src.data import load_data
from src.train import train
from src.output import output
import torch

def parse_args():
    '''Parses command line for input arguments'''
    parser = argparse.ArgumentParser(usage='main.py [--help] [-o results] [-n INT] [-d cuda] [-v True]')
    parser.add_argument("--output", "-o", help="Relative path to output directory")
    parser.add_argument("--number", "-n", help="Number of digit samples to generate")
    parser.add_argument("--device", '-d', help="device to run on: cuda or cpu")
    parser.add_argument("--verbose", '-v', help="True to turn on verbose output")
    return parser.parse_args()

def main(args):
    '''Main entry point for Assignment 5'''
    # don't change these values as it messes with the convolutions
    #   just here for clarity so you know what the model is using
    latent_size = 20
    kernal_size = 3
    torch.manual_seed(1234) # seeding for consistent results

    print("Using device: ", args.device)

    # model and data
    vae_model = VAE(latent_size, kernal_size).to(args.device)
    loader, input = load_data(args.device, batch_size=200)

    # training
    verbosity = True if args.verbose == "True" else False
    obj_vals = train(vae_model, loader, verbosity)

    # producing results
    output(vae_model, input, obj_vals, args.number, args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
