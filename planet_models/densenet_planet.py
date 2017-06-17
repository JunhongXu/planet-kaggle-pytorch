import torch.utils.model_zoo as model_zoo
from torchvision.models.densenet import model_urls
from torchvision.models import DenseNet


def densenet121(num_classes=17, pretrained=False):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), drop_rate=0.2,
                     num_classes=num_classes)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['densenet121'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'classifier' not in key})
        model.load_state_dict(model_dict)
    return model


def densenet169(num_classes=17, pretrained=False):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['densenet169'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'classifier' not in key})
        model.load_state_dict(model_dict)
    return model


def densenet161(num_classes=17, pretrained=False):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_classes)
    if pretrained:
        # load model dictionary
        model_dict = model.state_dict()
        # load pretrained model
        pretrained_dict = model_zoo.load_url(model_urls['densenet161'])
        # update model dictionary using pretrained model without classifier layer
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'classifier' not in key})
        model.load_state_dict(model_dict)
    return model


