from torchvision.models.inception import inception_v3, model_urls, model_zoo


def inception_v3_planet(pretrained=True):
    net = inception_v3(num_classes=17, aux_logits=False)
    if pretrained:
        state_dict = net.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        state_dict.update({key: pretrained_dict[key] for key in state_dict if 'fc' not in key})
        net.load_state_dict(state_dict)
    return net


if __name__ == '__main__':
    inception_v3_planet()