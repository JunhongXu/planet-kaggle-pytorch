from data.kgdataset import KgForestDataset
from torch.autograd import Variable
from baseline_ensembles import transforms, rotate180, rotate270, rotate90, default, verticalFlip, horizontalFlip, \
    mean, std
from util import predict
import numpy as np
from torch.utils.data import DataLoader
from planet_models.blender import Blender
from torchvision.transforms import Compose, Normalize, Lambda
from util import toTensor
import torch


def get_valid_loader():
    loader = KgForestDataset(
        split='train-40479',
        transform=Compose(
            [Lambda(lambda x: toTensor(x)), Normalize(mean=mean, std=std)]
        ),
        height=256,
        width=256
    )
    return DataLoader(loader, batch_size=256, shuffle=False)


def get_test_dataloader():
    test_dataset = KgForestDataset(
        split='test-61191',
        transform=Compose(
            [
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        label_csv=None
    )

    test_dataloader = DataLoader(test_dataset, batch_size=256)
    return test_dataloader


def pred_valid():
    net = Blender()
    net.load_state_dict(torch.load('models/full_data_blender_adam.pth'))
    net.eval()
    net.cuda()
    # preds = np.empty((len(transforms), valid_loader.dataset.images.shape[0], 17))
    imgs = valid_loader.dataset.images.copy()
    for t in transforms:
        name = 'blender'
        t_name = str(t).split()[1]
        print('[!]Transform: {}'.format(t_name))
        valid_loader.dataset.images = t(imgs)
        pred = predict(net, valid_loader)
        np.savetxt('probs/{}_{}.txt'.format(t_name, name), pred)


def find_threshold():
    pass


if __name__ == '__main__':
    valid_loader = get_valid_loader()
    pred_valid()