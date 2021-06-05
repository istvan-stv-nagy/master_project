import torch.nn as nn
import torch
import torch.nn.functional as F


class SegnetDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SegnetDownConv, self).__init__()
        self.conv = self.__make_layers(in_channels, out_channels, num_layers)

    def __make_layers(self, in_channels, out_channels, num_layers):
        layers = []
        layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        layers.append(layer_1)
        for i in range(0, num_layers - 1):
            layer_i = nn.Sequential(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            layers.append(layer_i)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SegnetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SegnetUpConv, self).__init__()
        self.conv = self.__make_layers(in_channels, out_channels, num_layers)

    def __make_layers(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(0, num_layers - 1):
            layer_i = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
            layers.append(layer_i)
        layer_n = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        layers.append(layer_n)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.encoder_0 = SegnetDownConv(in_channels=in_channels, out_channels=64, num_layers=2)
        self.encoder_1 = SegnetDownConv(in_channels=64, out_channels=128, num_layers=2)
        self.encoder_2 = SegnetDownConv(in_channels=128, out_channels=256, num_layers=3)
        self.encoder_3 = SegnetDownConv(in_channels=256, out_channels=512, num_layers=3)
        self.encoder_4 = SegnetDownConv(in_channels=512, out_channels=512, num_layers=3)

        self.decoder_4 = SegnetUpConv(in_channels=512, out_channels=512, num_layers=3)
        self.decoder_3 = SegnetUpConv(in_channels=512, out_channels=256, num_layers=3)
        self.decoder_2 = SegnetUpConv(in_channels=256, out_channels=128, num_layers=3)
        self.decoder_1 = SegnetUpConv(in_channels=128, out_channels=64, num_layers=2)
        self.decoder_0 = SegnetUpConv(in_channels=64, out_channels=64, num_layers=1)
        self.final_layer = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)

    def forward(self, x):
        dim_0 = x.size()
        x = self.encoder_0(x)
        x, i0 = self.max_pool(x)

        dim_1 = x.size()
        x = self.encoder_1(x)
        x, i1 = self.max_pool(x)

        dim_2 = x.size()
        x = self.encoder_2(x)
        x, i2 = self.max_pool(x)

        dim_3 = x.size()
        x = self.encoder_3(x)
        x, i3 = self.max_pool(x)

        dim_4 = x.size()
        x = self.encoder_4(x)
        x, i4 = self.max_pool(x)

        x = F.max_unpool2d(x, indices=i4, kernel_size=(2, 2), stride=(2, 2), output_size=dim_4)
        x = self.decoder_4(x)

        x = F.max_unpool2d(x, indices=i3, kernel_size=(2, 2), stride=(2, 2), output_size=dim_3)
        x = self.decoder_3(x)

        x = F.max_unpool2d(x, indices=i2, kernel_size=(2, 2), stride=(2, 2), output_size=dim_2)
        x = self.decoder_2(x)

        x = F.max_unpool2d(x, indices=i1, kernel_size=(2, 2), stride=(2, 2), output_size=dim_1)
        x = self.decoder_1(x)

        x = F.max_unpool2d(x, indices=i0, kernel_size=(2, 2), stride=(2, 2), output_size=dim_0)
        x = self.decoder_0(x)

        x = self.final_layer(x)
        return x


if __name__ == '__main__':
    model = SegNet(1, 1)
    xx = torch.randn(1, 1, 256, 256)
    yy = model(xx)
    print(yy.shape)