from torchvision.models.resnet import BasicBlock, ResNet, resnet34, resnet50, resnet101, resnet152, resnet18
from torch.nn import Module

def resnet34_planet():
    return resnet34(False, num_classes=17)


def resnet101_planet():
    return resnet101(pretrained=False, num_classes=17)


def resnet50_planet():
    return resnet50(pretrained=False, num_classes=17)


def resnet152_planet():
    return resnet152(pretrained=False, num_classes=17)


def resnet18_planet():
    return resnet18(pretrained=False, num_classes=17)


def resnet14_planet():
    resnet = ResNet(BasicBlock, [1, 2, 2, 1], num_classes=17)
    return resnet


