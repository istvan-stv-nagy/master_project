import torch.nn as nn
import torch.nn.functional as F


class InterChannelConvolutionNeuralNetwork(nn.Module):
    def __init__(self):
        super(InterChannelConvolutionNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=4, stride=2)
        self.conv3 = nn.Conv1d(4, 2, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(2, 1, kernel_size=4, stride=2)

        self.fc5 = nn.Linear(30, 12)
        self.fc6 = nn.Linear(12, 6)
        self.fc7 = nn.Linear(6, 2)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        #x = x.view(x.shape[0], x.shape[1])
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        return x
