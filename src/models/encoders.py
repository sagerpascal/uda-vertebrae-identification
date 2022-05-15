import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_blocks(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UnetIdentification(torch.nn.Module):

    def __init__(self, in_channels, filters=32, kernel_size=(3, 3)):
        super().__init__()
        self.down_1 = DownSamplingBlock2D(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                          with_pooling=False)
        self.down_2 = DownSamplingBlock2D(in_channels=filters, out_channels=2 * filters, kernel_size=kernel_size)
        self.down_3 = DownSamplingBlock2D(in_channels=2 * filters, out_channels=4 * filters, kernel_size=kernel_size)
        self.down_4 = DownSamplingBlock2D(in_channels=4 * filters, out_channels=8 * filters, kernel_size=kernel_size)
        self.floor = DownSamplingBlock2D(in_channels=8 * filters, out_channels=16 * filters, kernel_size=(5, 21),
                                         padding=(2, 10))
        self.up_4 = UpSamplingBlock2D(in_channels=16 * filters, skip_channels=self.down_4.out_channels,
                                      out_channels=8 * filters, kernel_size=kernel_size)
        self.up_3 = UpSamplingBlock2D(in_channels=8 * filters, skip_channels=self.down_3.out_channels,
                                      out_channels=4 * filters, kernel_size=kernel_size)
        self.up_2 = UpSamplingBlock2D(in_channels=4 * filters, skip_channels=self.down_2.out_channels,
                                      out_channels=2 * filters, kernel_size=kernel_size)
        self.up_1 = UpSamplingBlock2D(in_channels=2 * filters, skip_channels=self.down_1.out_channels,
                                      out_channels=filters, kernel_size=kernel_size)

        self.initialize()

    def initialize(self):
        initialize_blocks(self.down_1)
        initialize_blocks(self.down_2)
        initialize_blocks(self.down_3)
        initialize_blocks(self.down_4)
        initialize_blocks(self.floor)
        initialize_blocks(self.up_4)
        initialize_blocks(self.up_3)
        initialize_blocks(self.up_2)
        initialize_blocks(self.up_1)

    def forward(self, x):
        x_down_1 = self.down_1(x)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_4 = self.down_4(x_down_3)
        floor = self.floor(x_down_4)
        x_up_4 = self.up_4(floor, x_down_4)
        x_up_3 = self.up_3(x_up_4, x_down_3)
        x_up_2 = self.up_2(x_up_3, x_down_2)
        x_up_1 = self.up_1(x_up_2, x_down_1)

        return x_up_1

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class DownSamplingBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, with_pooling=True, padding=(1, 1)):
        super().__init__()
        self.out_channels = out_channels

        if with_pooling:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            self.pool = None

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act_2 = nn.ReLU()

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        return x


class UpSamplingBlock2D(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, stride=(1, 1),
                                padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act_2 = nn.ReLU()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        return x


class UnetDetection(torch.nn.Module):

    def __init__(self, in_channels, out_channels, filters=16, kernel_size=(3, 3, 3)):
        super().__init__()
        self.down_1 = DownSamplingBlock3D(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                          with_pooling=False)
        self.down_2 = DownSamplingBlock3D(in_channels=filters, out_channels=2 * filters, kernel_size=kernel_size)
        self.down_3 = DownSamplingBlock3D(in_channels=2 * filters, out_channels=4 * filters, kernel_size=kernel_size)
        self.down_4 = DownSamplingBlock3D(in_channels=4 * filters, out_channels=8 * filters, kernel_size=kernel_size)
        self.floor = DownSamplingBlock3D(in_channels=8 * filters, out_channels=16 * filters, kernel_size=kernel_size)
        self.up_4 = UpSamplingBlock3D(in_channels=16 * filters, skip_channels=self.down_4.out_channels,
                                      out_channels=8 * filters, kernel_size=kernel_size)
        self.up_3 = UpSamplingBlock3D(in_channels=8 * filters, skip_channels=self.down_3.out_channels,
                                      out_channels=4 * filters, kernel_size=kernel_size)
        self.up_2 = UpSamplingBlock3D(in_channels=4 * filters, skip_channels=self.down_2.out_channels,
                                      out_channels=2 * filters, kernel_size=kernel_size)
        self.up_1 = UpSamplingBlock3D(in_channels=2 * filters, skip_channels=self.down_1.out_channels,
                                      out_channels=filters, kernel_size=kernel_size)

        self.initialize()

    def initialize(self):
        initialize_blocks(self.down_1)
        initialize_blocks(self.down_2)
        initialize_blocks(self.down_3)
        initialize_blocks(self.down_4)
        initialize_blocks(self.floor)
        initialize_blocks(self.up_4)
        initialize_blocks(self.up_3)
        initialize_blocks(self.up_2)
        initialize_blocks(self.up_1)

    def forward(self, x):
        x_down_1 = self.down_1(x)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_4 = self.down_4(x_down_3)
        floor = self.floor(x_down_4)
        x_up_4 = self.up_4(floor, x_down_4)
        x_up_3 = self.up_3(x_up_4, x_down_3)
        x_up_2 = self.up_2(x_up_3, x_down_2)
        x_up_1 = self.up_1(x_up_2, x_down_1)
        return x_up_1

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def optim_parameters(self):
        for module in [self.down_1, self.down_1, self.down_1, self.down_1, self.down_1, self.floor, self.up_1,
                       self.up_2, self.up_3, self.up_4, self.up_6, self.out_conv]:
            for param in module.parameters():
                yield param


class DownSamplingBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, with_pooling=True):
        super().__init__()
        self.out_channels = out_channels

        if with_pooling:
            self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        else:
            self.pool = None

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn_1 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.bn_2 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.act_2 = nn.ReLU()

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        return x


class UpSamplingBlock3D(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.bn_1 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.bn_2 = nn.BatchNorm3d(out_channels, momentum=0.1)
        self.act_2 = nn.ReLU()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        return x
