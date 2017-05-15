from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18


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

