import torch
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'models'))

from models import ConvolutionalNetwork

for path in sys.path:
    print(path)

def train():
    model = ConvolutionalNetwork()
    print(model)

if __name__ == '__main__':
    train()

