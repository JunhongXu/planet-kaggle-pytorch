import torch.nn as nn
import math
from torch.nn import functional as F
import torch
from torchvision.models.resnet import model_zoo, model_urls


def fpn_50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)
    return model


def fpn_152(pretrained=True):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=17)
    if pretrained:
        state_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        pretrained_dict = {key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
    return model


def fpn_34(pretrained=True):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=17)
    if pretrained:
        state_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_dict = {key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
    return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # 64*64*64
        # bottom-up
        self.layer1 = self._make_layer(block, 64, layers[0])                # 256*64*64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # 512*32*32
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # 1024*16*16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)     # 2048*8*8
        self.avgpool = nn.AvgPool2d(7)                                      # 2048*1*1

        # top-down
        self.td_1 = self._make_top_down_layer(512*block.expansion, 256*block.expansion)                     # 1024*8*8
        self.td_2 = self._make_top_down_layer(256*block.expansion, 128*block.expansion)                    # 512*16*16
        self.td_3 = self._make_top_down_layer(128*block.expansion, 64*block.expansion)                     # 256*32*32

        # extra conv layers
        self.p1_conv = self._make_conv_bn(256*block.expansion, 256, 3, padding=0, stride=2, use_relu=True)                # 256*4*4
        self.p2_conv = self._make_conv_bn(256*block.expansion, 256, 3, padding=0, stride=2, use_relu=True)                # 256*8*8
        self.p3_conv = self._make_conv_bn(128*block.expansion, 256, 3, padding=0, stride=2, use_relu=True)                # 256*16*16
        self.p4_conv = self._make_conv_bn(64*block.expansion, 256, 3, padding=0, stride=2, use_relu=True)                 # 256*32*32

        # classification layer
        self.cls1 = nn.Linear(256, out_features=num_classes, bias=True)
        self.cls2 = nn.Linear(2304, out_features=num_classes, bias=True)
        self.cls3 = nn.Linear(12544, out_features=num_classes, bias=True)
        self.cls4 = nn.Linear(57600, out_features=num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_top_down_layer(self, inpt_size, out_size):
        # lateral connection
        x = self._make_conv_bn(inpt_size=inpt_size, out_size=out_size, k_size=1,
                                     padding=0, stride=1, use_relu=True)

        return x

    def _make_conv_bn(self, inpt_size, out_size, k_size, padding=1, stride=1, use_relu=False):
        x = nn.Sequential(
            nn.Conv2d(in_channels=inpt_size, bias=False, out_channels=out_size,
                      kernel_size=k_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_size)
        )
        if use_relu:
            x.add_module('relu', nn.ReLU())
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        N = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())

        layer_1 = self.layer1(x)
        # print(layer_1.size())

        layer_2 = self.layer2(layer_1)
        # print(layer_2.size())

        layer_3 = self.layer3(layer_2)
        # print(layer_3.size())

        layer_4 = self.layer4(layer_3)
        # print(layer_4.size())

        p_1 = self.td_1(layer_4)
        # print(p_1.size())
        upsample_p1 = F.upsample_nearest(p_1, scale_factor=2)

        p_2 = upsample_p1 + layer_3
        # print(p_2.size())
        upsample_p2 = F.upsample_nearest(p_2, scale_factor=2)

        p_3 = self.td_2(upsample_p2) + layer_2
        # print(p_3.size())
        upsample_p3 = F.upsample_nearest(p_3, scale_factor=2)

        p_4 = self.td_3(upsample_p3) + layer_1
        # print(p_4.size())
        p_1 = self.p1_conv(p_1)
        p_2 = self.p2_conv(p_2)

        p_3 = self.p3_conv(p_3)
        p_4 = self.p4_conv(p_4)

        # pooling
        cls_1 = self.cls1(F.avg_pool2d(p_1, kernel_size=3).view(N, -1))
        cls_2 = self.cls2(F.avg_pool2d(p_2, kernel_size=3, stride=2).view(N, -1))
        cls_3 = self.cls3(F.avg_pool2d(p_3, kernel_size=3, stride=2).view(N, -1))
        cls_4 = self.cls4(F.avg_pool2d(p_4, kernel_size=3, stride=2).view(N, -1))
        # concatenate
        # cls = torch.cat([cls_1, cls_2, cls_3, cls_4], 1)
        cls = cls_1 + cls_2 + cls_3 + cls_4
        # print(cls.size())
        cls = cls.view(cls.size(0), -1)
        # x = self.fc(cls)
        return cls


if __name__ == '__main__':
    from torch.autograd import Variable
    net =fpn_34()
    x = Variable(torch.randn(1, 3, 256, 256))
    net(x)