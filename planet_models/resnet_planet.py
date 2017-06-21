from torchvision.models.resnet import BasicBlock, ResNet, resnet34, resnet50, resnet101, resnet152, \
    resnet18, model_urls, model_zoo
import torch.nn as nn
import math


def resnet34_planet(pretrained=False):
    model = resnet34(False, num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)

    return model


def resnet101_planet(pretrained=False):
    model = resnet101(False, num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)

    return model


def resnet50_planet(pretrained=False):
    model = resnet50(False, num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)

    return model


def resnet152_planet(pretrained=False):
    model = resnet152(False, num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)

    return model


def resnet18_planet(pretrained=False):
    model = resnet18(False, num_classes=17)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        model.load_state_dict(model_dict)

    return model


def resnet14_planet():
    resnet = ResNet(BasicBlock, [1, 2, 2, 1], num_classes=17)
    return resnet


def resnet10_planet():
    return CustomizedResNet(BasicBlock, [1, 1, 1, 1], num_classes=17)


def resnet14_nrgb():
    resnet = CustomizedResNet(BasicBlock, [1, 2, 2, 1], num_channels=4, num_classes=17)
    return resnet


class CustomizedResNet(nn.Module):
    def __init__(self, block, layers, num_channels=3, num_classes=1000):
        self.inplanes = 64
        super(CustomizedResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.dropout3 = nn.Dropout2d(p=0.3)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



