import torch.nn as nn
from torch.nn import functional as F
import math
from collections import OrderedDict
import torch
from torch.autograd import Variable


def _make_conv_bn_elu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return OrderedDict([
        ('conv2d', nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, bias=False)),
        ('batch_norm', nn.BatchNorm2d(out_channels)),
        ('elu', nn.ELU(inplace=True))
    ])


def _make_linear_bn_elu(in_units, output_units):
    return OrderedDict(
            [
                ('linear', nn.Linear(in_units, output_units, bias=False)),
                ('batch_norm', nn.BatchNorm1d(output_units)),
                ('elu', nn.ELU()),
            ]
        )


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
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


class FPNet(nn.Module):

    def __init__(self, block, layers, input_channels=3, num_classes=17):
        self.inplanes = 32
        super(FPNet, self).__init__()
        self.conv1 = nn.Sequential(
            _make_conv_bn_elu(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=0)
        )                                                                            # 35*35*3
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)                         # 32*32*36

        # downsampling
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)               # 18*18*128
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)              # 8*8*256
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)              # 4*4*512
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)              # 2*2*1024
        # downsampling prediction
        self.d_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_d = nn.Sequential(_make_linear_bn_elu(in_units=1024, output_units=512))

        # upsampling
        self.layer3_up = nn.Sequential(_make_conv_bn_elu(self.inplanes//2, self.inplanes,
                                                         kernel_size=1, stride=1, padding=0))    # 4*4*1024
        self.layer2_up = nn.Sequential(_make_conv_bn_elu(self.inplanes//4, self.inplanes//2,
                                                         kernel_size=1, stride=1, padding=0))  # 8*8*512
        self.layer1_up = nn.Sequential(_make_conv_bn_elu(self.inplanes//8, self.inplanes//4,
                                                         kernel_size=1, stride=1, padding=0) ) # 16*16*256

        # final feature generation
        self.f1 = nn.Sequential(_make_conv_bn_elu(256, 512, kernel_size=3, stride=2))  # 8*8*512
        self.f2 = nn.Sequential(_make_conv_bn_elu(512, 512, kernel_size=3, stride=2))  # 4*4*512
        self.f3 = nn.Sequential(_make_conv_bn_elu(1024, 512))                          # 4*4*512

        # reduce dimensionality before classifier 1*1*256
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        # clasifier
        self.cls_1 = nn.Sequential(_make_linear_bn_elu(512, 512))
        self.cls_2 = nn.Sequential(_make_linear_bn_elu(512, 512))
        self.cls_3 = nn.Sequential(_make_linear_bn_elu(512, 512))

        # logit
        self.fc = nn.Linear(512*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.conv1(x)
        x = self.maxpool(x)     # 32*32*36 now using kernel 10*10, maybe smaller kernel?
        # downsampling
        d1 = self.layer1(x)     # 16*16*128
        d2 = self.layer2(d1)     # 8*8*256
        d3 = self.layer3(d2)     # 4*4*512
        d4 = self.layer4(d3)     # 2*2*1024

        # upsampling, will using transpose conv2d better?
        up1 = F.upsample_nearest(d4, scale_factor=2)    # 4*4*1024
        up2 = F.upsample_nearest(d3, scale_factor=2)    # 8*8*512
        up3 = F.upsample_nearest(d2, scale_factor=2)    # 16*16*256

        # lateral connection
        l1 = self.layer1_up(d1)     # 16*16*256
        l2 = self.layer2_up(d2)     # 8*8*512
        l3 = self.layer3_up(d3)     # 4*4*1024

        # merge
        m1 = l1 + up3   # 16*16*256
        m2 = l2 + up2   # 8*8*512
        m3 = l3 + up1   # 4*4*1024

        # extra convs
        f1 = self.f1(m1)    # 8*8*256
        f2 = self.f2(m2)    # 4*4*256
        f3 = self.f3(m3)    # 2*2*256

        # max pool
        f1 = self.pool1(f1)  # 512
        f1 = f1.view(f1.size(0), -1)
        f2 = self.pool2(f2)  # 512
        f2 = f2.view(f2.size(0), -1)
        f3 = self.pool3(f3)  # 512
        f3 = f3.view(f3.size(0), -1)

        # downsampling classifier
        d_out = self.d_pool(d4)
        d_out = d_out.view(d_out.size(0), -1)
        d_out = self.cls_d(d_out)   # 512

        # classifier
        cls1 = self.cls_1(f1)   # 512
        cls2 = self.cls_2(f2)   # 512
        cls3 = self.cls_3(f3)   # 512

        cls = torch.cat((cls1, cls2, cls3, d_out), dim=1)

        logit = self.fc(cls)

        prob = F.sigmoid(logit)
        return logit, prob


if __name__ == '__main__':
    import time
    x = torch.randn((1, 3, 72, 72))
    x = Variable(x).cuda()
    net = FPNet(Bottleneck, [2, 2, 2, 2])
    net.cuda()
    t = time.time()
    net(x)
    print(time.time() - t)
