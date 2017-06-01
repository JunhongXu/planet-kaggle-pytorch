import torch.nn as nn
import torch
import math

"""
A 17 layer network
"""


class Block(nn.Module):
    def __init__(self, inpt_channel, output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inpt_channel, kernel_size=1, out_channels=output_channel//2, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel//2)
        self.elu1 = nn.ELU()

        self.conv2 = nn.Conv2d(in_channels=output_channel//2, out_channels=output_channel//2, bias=False,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel//2)
        self.elu2 = nn.ELU()

        self.conv3 = nn.Conv2d(in_channels=output_channel//2, out_channels=output_channel, kernel_size=1, bias=False,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.elu3 = nn.ELU()

        self.stage = nn.Sequential(
            self.conv1, self.bn1, self.elu1,
            self.conv2, self.bn2, self.elu2, nn.Dropout2d(0.2),
            self.conv3, self.bn3, self.elu3
        )

    def forward(self, x):
        return self.stage(x)


class TransitionBlock(nn.Module):
    def __init__(self, size, input_channel, output_channel=2048):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, stride=1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        return x


class SimpleNetV3(nn.Module):
    def __init__(self, inpt_size=72, input_channel=3):
        super(SimpleNetV3, self).__init__()
        self.pre_layer = nn.Sequential(
            self._make_conv_bn_elu(3, 16),
            self._make_conv_bn_elu(16, 16),
            self._make_conv_bn_elu(16, 16)
        )
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (64, 36, 36)
        self.stage_1 = Block(inpt_channel=64, output_channel=256)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.transition_0 = TransitionBlock(18, 256, 2048)
        # (256, 18, 18)
        self.stage_2 = Block(inpt_channel=256, output_channel=512)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transition_1 = TransitionBlock(8, 512)
        # (512, 8, 8)
        self.stage_3 = Block(inpt_channel=512, output_channel=1024)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transition_2 = TransitionBlock(4, 1024)
        # (1024, 4, 4)
        self.stage_4 = Block(inpt_channel=1024, output_channel=2048)
        self.avg_pool = nn.AvgPool2d(4)
        # (2048, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2048*3, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.5),
               nn.Linear(512, 17)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_bn_elu(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, padding=0, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        return layer

    def forward(self, x):
        x = self.pre_layer(x)

        x = self.conv1(x)
        x = self.maxpool_1(x)

        # stage 1
        x = self.stage_1(x)
        x = self.maxpool_1(x)
        # transition_0 = self.transition_0(x)

        # stage 2
        x = self.stage_2(x)
        x = self.maxpool_2(x)
        transition_1 = self.transition_1(x)

        # stage 3
        x = self.stage_3(x)
        x = self.maxpool_3(x)
        transition_2 = self.transition_2(x)

        # stage 4
        x = self.stage_4(x)
        x = self.avg_pool(x)
        x = torch.cat([x, transition_1, transition_2], -1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
