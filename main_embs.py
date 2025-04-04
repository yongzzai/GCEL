'''
@author: Y.J.Lee
'''

import os
import argparse

parser = argparse.ArgumentParser(description='run')
parser.add_argument('-hd', '--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('-nl', '--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('-l', '--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
args = parser.parse_args()


import torch
from utils.fs import EVENTLOG_DIR
from dataset.dataset import Dataset
from model.GCEL import GCEL

def main():

    dataset_names = [name.split('.')[0] for name in os.listdir(EVENTLOG_DIR) if name.endswith('.csv')]

    for d in dataset_names:

        print(f'current dataset: {d}')
        dataset = Dataset(d)

        torch.manual_seed(1999)
        model = GCEL(**vars(args))
        model.fit(dataset)
        model.visualize()


if __name__ == '__main__':
    main()

