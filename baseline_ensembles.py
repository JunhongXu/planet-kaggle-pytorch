import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.kgdataset import KgForestDataset, toTensor
from torchvision.transforms import Normalize, Compose, Lambda
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet
from planet_models.densenet_planet import densenet161, densenet121, densenet169
from util import predict


def default(imgs):
    return imgs


def rotate90(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = cv2.transpose(img, (1, 0, 2))
    return imgs


def rotate180(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = cv2.flip(img, -1)
    return imgs


def rotate270(imgs):
    for index, img in enumerate(imgs):
        img = cv2.transpose(img, (1, 0, 2))
        imgs[index] = cv2.flip(img, -1)
    return imgs


def horizontalFlip(imgs):
    for index, img in enumerate(imgs):
        img = cv2.flip(img, 1)
        imgs[index] = img
    return imgs


def verticalFlip(imgs):
    for index, img in enumerate(imgs):
        img = cv2.flip(img, 0)
        imgs[index] = img
    return imgs


mean = [0.31151703, 0.34061992, 0.29885209]
std = [0.16730586, 0.14391145, 0.13747531]
transforms = [default, rotate90, rotate180, rotate270, verticalFlip, horizontalFlip]
models = [resnet18_planet, resnet34_planet, resnet50_planet, densenet121, densenet161, densenet169]




# if __name__ == '__main__':
#     img = cv2.imread('images.jpeg')
#     img = cv2.resize(img, (256, 256))
#     img = np.expand_dims(img, axis=0)
#     rotation90 = rotate90(img.copy())[0]
#     rotation180 = rotate180(img.copy())[0]
#     rotation270 = rotate270(img.copy())[0]
#     vertical = verticalFlip(img.copy())[0]
#     horizontal = horizontalFlip(img.copy())[0]
#     cv2.imshow('original', img[0])
#     cv2.imshow('90', rotation90)
#     cv2.imshow('180', rotation180)
#     cv2.imshow('270', rotation270)
#     cv2.imshow('vertical', vertical)
#     cv2.imshow('horizontal', horizontal)
#
#     cv2.waitKey()


# save probabilities to files for debug
def probs(dataloader):
    """
    returns a numpy array of probabilities (n_transforms, n_models, n_imgs, 17)
    use transforms to find the best threshold
    use models to do ensemble method
    """
    n_transforms = len(transforms)
    n_models = len(models)
    n_imgs = dataloader.dataset.num
    imgs = dataloader.dataset.images.copy()
    probabilities = np.empty((n_transforms, n_models, n_imgs, 17))
    for t_idx, transform in enumerate(transforms):
        t_name = str(transform).split()[1]
        dataloader.dataset.images = transform(imgs)
        for m_idx, model in enumerate(models):
            name = str(model).split()[1]
            net = model().cuda()
            net = nn.DataParallel(net)
            net.load_state_dict(torch.load('models/{}.pth'.format(name)))
            net.eval()
            # predict
            m_predictions = predict(net, dataloader)

            # save
            np.savetxt(X=m_predictions, fname='probs/{}_{}.txt'.format(t_name, name))
            probabilities[t_idx, m_idx] = m_predictions
    return probabilities

# average the results from [verticalFlip, horizontalFlip, transpose]

# optimize the results


if __name__ == '__main__':
    validation = KgForestDataset(
        split='validation-3000',
        transform=Compose(
            [
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )
    valid_dataloader = DataLoader(validation, batch_size=256, shuffle=False)
    print(probs(valid_dataloader))
