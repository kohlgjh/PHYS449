import argparse
from src.data_load import load_data
from src.ising import Ising1D
import matplotlib.pyplot as plt
import numpy as np

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
    KLs = model.train_weights(input_data, 250, 0.05, flips_per_site=25, verbose=verbosity)

    # calcualte average KL over every 10 epochs
    avg_KL = np.average(np.array(KLs).reshape(-1, 10), axis=1)

    # plotting KLs
    plt.style.use('default')
    plt.plot(np.arange(0, 250), KLs, label='KL raw')
    plt.plot(np.arange(0, 250, 10), avg_KL, label='Averaged KL')
    plt.xlabel('Epoch')
    plt.ylabel('Estimated KL divergence')
    plt.title('KL Divergence Estiamte During Training')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('KL_plot.png', dpi=150)

if __name__ == '__main__':
    args = parse_args()
    main(args)
