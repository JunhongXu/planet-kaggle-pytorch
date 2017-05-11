"""
    A re-implementation of ResNeXT. All the blocks are of bottleneck type.
    The code follows the style of resnet.py in pytorch vision model.
"""
import torch
from torch.nn import *
from torch.nn import functional as F


class Bottleneck(Module):
    """Type C in the paper"""
    def __init__(self, inplanes, planes, cardinality, stride, downsample=None, activation_fn=ELU):
        """
        Parameters:
            inplanes: # of input channels
            planes: # of output channels
            cardinality: # of convolution groups
            stride: convolution stride
            downsample: convolution operation to increase the width of the output
            activation_fn: activation function
        """
        super(Bottleneck, self).__init__()
        depth = planes/2
        self.conv1 = Conv2d(inplanes, depth, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(planes/2)
        # group convolution
        self.conv2 = Conv2d(depth, depth, kernel_size=3, groups=cardinality, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes/2)
        # increase depth
        self.conv3 = Conv2d(planes/2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(planes)

    def forward(self, x):
        pass


class ResNeXT(Module):
    def __init__(self, block, depths, num_classes, cardinality=32, activation_fn=ELU):
        super(ResNeXT, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.activation = activation_fn(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layers(block, self.inplanes, depths[0])
        self.stage2 = self._make_layers(block, self.inplanes, depths[1], stride=2)
        self.stage3 = self._make_layers(block, self.inplanes, depths[2], stride=2)
        self.stage4 = self._make_layers(block, self.inplanes, depths[3], stride=2)
        self.avgpool = AvgPool2d(7)
        self.fc = Linear(block.expansion*512, num_classes)

    def _make_layers(self, block, planes, blocks, stride=1):
        pass

    def forward(self, x):
        pass
