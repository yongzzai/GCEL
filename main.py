'''
@author: Y.J.Lee
'''


import argparse

#parser = argparse.ArgumentParser(description='run')
#parser.add_argument('-r','--runtype', type=str, required=True, help="one of -> 'create_graphs', 'train', 'test', 'predict'")
#args = parser.parse_args()

import dataset.dataset as ds
dd = ds.Dataset('BPIC13_closed_problems')

import torch
from model.GCEL import GCEL

torch.manual_seed(1999)
model = GCEL(epochs=20, batch_size=256, lr=0.001)

model.fit(dd)
model.visualize()