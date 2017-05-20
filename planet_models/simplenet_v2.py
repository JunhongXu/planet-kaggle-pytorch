import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inpt_channel, output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inpt_channel, kernel_size=1, out_channels=output_channel/2,
                               padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel/2)
        self.elu1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=output_channel/2, out_channels=output_channel/2,
                               kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel/2)
        self.elu2 = nn.ELU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=output_channel/2, out_channels=output_channel, kernel_size=1, stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.elu3 = nn.ELU(inplace=True)

        self.stage = nn.Sequential(
            self.conv1, self.bn1, self.elu1,
            self.conv2, self.bn2, self.elu2,
            self.conv3, self.bn3, self.elu3
        )

    def forward(self, x):
        return self.stage(x)


class SimpleNetV2(nn.Module):
    def __init__(self, inpt_size=64, input_channel=3):
        super(SimpleNetV2, self).__init__()

    def forward(self, x):
        pass
