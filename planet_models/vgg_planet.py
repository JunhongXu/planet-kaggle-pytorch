from torchvision.models.vgg import vgg19, model_urls, model_zoo


def vgg19_bn_planet(pretrained=True):
    net = vgg19(num_classes=17)
    if pretrained:
        state_dict = net.state_dict()
        # load pretrained dictionary
        pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
        state_dict.update({key: pretrained_dict[key] for key in state_dict if 'classifier' not in key})
        net.load_state_dict(state_dict)
    return net


if __name__ == '__main__':
    vgg19_bn_planet()