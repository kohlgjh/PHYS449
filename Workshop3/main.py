# -----------------------------------------------------------------------------
#
# Phys 449--Fall 2022
# Workshop 3 -- CIFAR10 example with tensorboard
#
# -----------------------------------------------------------------------------
import argparse, torch, sys
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# torch -- tensorboard interface
from torch.utils.tensorboard import SummaryWriter 

sys.path.append('src')

from nn_gen import Net
from load_data import CIFAR10Data
import image_displaying as img

def prepare(batch_size):
    data = CIFAR10Data(batch_size, download=True)
    num_classes = len(data.classes)
    net = Net(num_classes)

    return net, data

def train(args):
    lr = args.lr
    batch_size = args.batch_size
    res_path = args.res_path
    epochs = args.epochs

    net, data_loader = prepare(batch_size)
    train_loader, test_loader = data_loader.train_loader, data_loader.test_loader
    loss = nn.CrossEntropyLoss()

    assert args.sgd != args.adam, 'Choose SGD or Adam!'
    if args.sgd:
        optimizer = optim.SGD(net.parameters(), lr=lr)
        opt_tag = 'SGD'
    elif args.adam:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt_tag = 'Adam'

    run_ID = 'lr={}_batch_size={}_opt='.format(lr, batch_size) + opt_tag
    writer = SummaryWriter(res_path + run_ID)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs, labels = data

            outputs = net(inputs)
            obj_val = loss(outputs, labels)
            obj_val.backward()
            optimizer.step()

            running_loss += obj_val.item()

        # uses the last "inputs" and "labels" from the above loop
        writer.add_figure(
            'prediction vs. actual',
            img.plot_classes_preds(
                net, inputs, labels, data_loader.classes
            ),
            global_step=epoch
        )

        writer.add_scalar(
            'training loss',
            running_loss / (i+1), # i+1 = number of updates in 1 epoch
            epoch
        )

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser(description='CIFAR10 example with Tensorboard')

    # Could put these args into a json file...
    parser.add_argument('--lr', type=float, metavar='lr',
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, metavar='batch_size',
                        help='Batch size')
    parser.add_argument('--epochs', type=int, metavar='epochs',
                        help='Number of training epochs')
    parser.add_argument('--res-path', type=str,
                        help='Path to write runs to')
    parser.add_argument('--sgd', action='store_true',
                        help='Use SGD')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam')

    args = parser.parse_args()
    train(args)