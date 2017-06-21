"""
    A re-implementation of ResNeXT. All the blocks are of bottleneck type.
    The code follows the style of resnet.py in pytorch vision model.
"""
import torch
from torch.nn import *
from torch.autograd import Variable
from torch.nn.init import kaiming_normal


class Bottleneck(Module):
    expansion = 4
    """Type C in the paper"""
    def __init__(self, inplanes, planes, width, cardinality, stride=1, downsample=None, activation_fn=ELU(inplace=True)):
        """
        Params:
            inplanes: # of input channels
            planes: # of output channels
            width: # of channels in the bottleneck layer
            cardinality: # of convolution groups
            stride: convolution stride
            downsample: convolution operation to increase the width of the output
            activation_fn: activation function
        """
        super(Bottleneck, self).__init__()
        d = width * cardinality
        # reduce width
        self.conv1 = Conv2d(inplanes, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(d)
        # group convolution
        self.conv2 = Conv2d(d, d, kernel_size=3,  groups=cardinality, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(d)
        # increase width
        self.conv3 = Conv2d(d, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.downsample = downsample
        self.activation = activation_fn

    def forward(self, x):
        residual = x

        # reduce width
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # group conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        # increase width
        out = self.conv3(out)
        out = self.bn3(out)

        # identity mapping or projection
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.activation(out)
        return out


class ResNeXT(Module):
    def __init__(self, block, depths, width, num_classes, cardinality=32, activation_fn=ELU(inplace=True)):
        super(ResNeXT, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.width = width
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.activation = activation_fn
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layers(block, 256, 'stage1', depths[0])
        self.stage2 = self._make_layers(block, 512, 'stage2', depths[1], stride=2)
        self.stage3 = self._make_layers(block, 1024, 'stage3', depths[2], stride=2)
        self.stage4 = self._make_layers(block, 2048, 'stage4', depths[3], stride=2)
        self.avgpool = AvgPool2d(7)
        self.fc = Linear(block.expansion*512, num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, name, blocks, stride=1):
        """
        Params
            block: type of the ResNeXT block
            planes: output channels
            blocks: number of residual blocks in this stage
            name: name of this stage
            stride: stride for the first residual block of each stage
        Returns
            A sequential module representing the stage
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential(
                Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                BatchNorm2d(planes)
            )

        stage = Sequential()
        # first residual block uses projection
        stage.add_module('{}_block1'.format(name), block(self.inplanes, planes,  self.width, self.cardinality, stride,
                                                         downsample, self.activation))
        self.inplanes = planes
        for i in range(1, blocks):
            stage.add_module('{}_block{}'.format(name, i+1), block(self.inplanes, planes, self.width, self.cardinality,
                                                                   activation_fn=self.activation))
        self.width *= 2
        return stage

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.maxpool(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnext_35(num_labels=17):
    return ResNeXT(Bottleneck, [2, 3, 4, 2], 4, num_labels)


def resnext_29(num_labels=17):
    return ResNeXT(Bottleneck, [2, 2, 3, 2], 4, num_labels)


def resnext_11(num_labels=17):
    return ResNeXT(Bottleneck, [1, 1, 1, 1], 4, num_labels)
