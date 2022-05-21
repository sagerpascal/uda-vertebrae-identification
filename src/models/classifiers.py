import torch.nn as nn


class DRNSegPixelClassifier2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DRNSegPixelClassifier2D, self).__init__()
        self.f_block_5 = ForwardBlock2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        activation=None, dropout=False)

    def forward(self, x):
        x = self.f_block_5(x)
        return x


class ForwardBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', dropout=True):
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn_1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        if dropout:
            self.dropout_1 = nn.Dropout(0.2)
        if self.activation == 'relu':
            self.act_1 = nn.ReLU()
        else:
            self.act_1 = None

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        if self.activation is not None and self.act_1 is not None:
            x = self.act_1(x)
        if self.dropout:
            x = self.dropout_1(x)
        return x


class DRNSegPixelClassifier3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DRNSegPixelClassifier3D, self).__init__()
        self.f_block_5 = ForwardBlock3D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        activation=None, dropout=False)

    def forward(self, x):
        x = self.f_block_5(x)
        return x


class ForwardBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', dropout=True):
        super().__init__()
        self.activation = activation
        self.dropout = dropout
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn_1 = nn.BatchNorm3d(out_channels, momentum=0.1)
        if dropout:
            self.dropout_1 = nn.Dropout(0.2)
        if self.activation == 'relu':
            self.act_1 = nn.ReLU()
        else:
            self.act_1 = None

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        if self.activation is not None and self.act_1 is not None:
            x = self.act_1(x)
        if self.dropout:
            x = self.dropout_1(x)
        return x
