from torchvision.models.inception import inception_v3, model_urls, model_zoo


def inception_v3_planet(pretrained=True):
    net = inception_v3(num_classes=17)
    if pretrained:
        state_dict = net.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        pretrained_dict.update({key: pretrained_dict[key] for key in state_dict if 'fc' not in state_dict})
        net.load_state_dict(pretrained_dict)
    return net