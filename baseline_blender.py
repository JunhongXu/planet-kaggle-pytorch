from data.kgdataset import KgForestDataset
from torch.autograd import Variable
from baseline_ensembles import transforms, rotate180, rotate270, rotate90, default, verticalFlip, horizontalFlip, \
    mean, std
from thresholds import thresholds
import glob
from util import predict, f2_score
import numpy as np
from torch.utils.data import DataLoader
from planet_models.blender import Blender
from torchvision.transforms import Compose, Normalize, Lambda
from util import toTensor, pred_csv
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
    threshold = np.zeros(17)
    acc  = 0
    labels = valid_loader.dataset.labels
    pred_files = [f for f in glob.glob('probs/*.txt') if 'blender' in f]
    for f in pred_files:
        selected_preds = np.loadtxt(f)
        t = np.ones(17) * 0.15
        # selected_preds = probabilities[t_idx]
        # selected_preds = np.mean(selected_preds, axis=0)
        best_thresh = 0.0
        best_score = 0.0
        for i in range(17):
            for r in range(500):
                r /= 500
                t[i] = r
                preds = (selected_preds > t).astype(int)
                score = f2_score(labels, preds)
                if score > best_score:
                    best_thresh = r
                    best_score = score
            t[i] = best_thresh
        threshold = threshold + t
        acc += best_score
    print('AVG ACC,', acc / len(pred_files))
    threshold = threshold / len(pred_files)
    return threshold


def pred_test():
    imgs = test_loader.dataset.images
    for idx, t in enumerate(transforms):
        print('[!]Transforms {}'.format(str(t).split()[1]))
        test_loader.dataset.images = t(imgs)
        preds = predict(net, test_loader)
        pred_labels = (preds > thresholds['blender']).astype(int)
        np.savetxt('submission_probs/full_data_{}_blender.txt'.format(str(t).split()[1]), pred_labels)


def test_majority_blender():
    label_files = [f for f in glob.glob('submission_probs/*.txt') if 'blender' in f]
    label = np.zeros((61191, 17))
    for f in label_files:
        l = np.loadtxt(f)
        label = l + label
    label = (label >= len(transforms)).astype(int)
    pred_csv(label, 'blender')


if __name__ == '__main__':
    net = Blender()
    net.load_state_dict(torch.load('models/full_data_blender_adam.pth'))
    net.eval()
    net.cuda()

    valid_loader = get_valid_loader()
    pred_valid()
    print(list(find_threshold()))
    # test_loader = get_test_dataloader()
    # test_majority_blender()
