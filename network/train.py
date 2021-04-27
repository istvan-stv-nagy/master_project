import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.curb_dataset import CurbstoneDataset
from network.chnn import ChannelNeuralNetwork
from network.fcnn import FullyConnectedNeuralNetwork


def train(trainset, net: nn.Module, epochs=3):
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)
    return net


if __name__ == '__main__':
    dataset = CurbstoneDataset()
    net = FullyConnectedNeuralNetwork()
    Data = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4)
    net = train(Data, net, epochs=12)
    torch.save(net.state_dict(), r'E:\Storage\7 Master Thesis\results\network\model_fcnn')

