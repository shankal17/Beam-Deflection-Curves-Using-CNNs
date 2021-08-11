import sys
import os
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'scripts'))
sys.path.insert(0, os.path.join(parentdir, 'models'))
# sys.path.insert(0, os.path.join(parentdir, 'dataset'))

from models import ConvAutoencoder1d
from dataset import BeamDataset2d


def train_2d_problem(model, train_loader, epochs=20, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Training on ' + str(list(model.parameters())[0].device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for locations, forces, deflections in tqdm(train_loader):
            locations = locations.to(device)
            forces = forces.to(device)
            deflections = deflections.to(device)

            # Zero optimizer gradients
            optimizer.zero_grad()

            # Predict deflection curve
            predicted_deflections = locations * model(forces) #TODO: experiment

            # Compute loss
            loss = criterion(predicted_deflections, deflections)

            # Backpropagate
            loss.backward()

            # Update model
            optimizer.step()

            # increment running loss
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_loader)
        print('epoch: {} Loss {:.6f}'.format(epoch, epoch_loss))
    
    return model


if __name__ == '__main__':
    # train()
    data = BeamDataset2d('data/test', 'test')
    data_loader = torch.utils.data.DataLoader(data, batch_size=8)
    data_iterator = iter(data_loader)
    x, forces, deflection = data_iterator.next()
    print(x.shape)
    # print(data.__getitem__(0)[1])

