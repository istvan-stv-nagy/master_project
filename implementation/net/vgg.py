import torch
import torch.nn as nn


VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, in_channels, resolution, num_classes, architecture_name='VGG16'):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        architecture = VGG_types[architecture_name]
        self.conv_layers = self.create_conv_layers(architecture)
        max_pool_count = architecture.count('M')
        downsample = 2 ** max_pool_count
        final_downsampled_size = (resolution[0] // downsample) * (resolution[1] // downsample)
        self.fcs = nn.Sequential(
            nn.Linear(512*final_downsampled_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                ]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VGG(in_channels=3, resolution=(244, 244), num_classes=1000, architecture_name="VGG19")
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)