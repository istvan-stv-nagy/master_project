import torch
from torch.utils.data import Dataset, DataLoader
from dataset.dataset_reader import read_curbstone_dataset
from network.fcnn import FullyConnectedNeuralNetwork


class CurbstoneDataset(Dataset):
    def __init__(self):
        x, y = read_curbstone_dataset()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]
        print("samples=", self.n_samples)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
