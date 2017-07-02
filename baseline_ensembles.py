import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.kgdataset import KgForestDataset, toTensor
from torchvision.transforms import Normalize, Compose, Lambda
import glob
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet
from planet_models.densenet_planet import densenet161, densenet121, densenet169
from util import predict, f2_score, pred_csv
from data import kgdataset



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

thresholds = [
#     [0.22433333, 0.20966667, 0.17,
#      0.07, 0.20433333, 0.23033333,
#      0.156, 0.23033333, 0.19933333,
#      0.21533333, 0.13, 0.156, 0.205,
#      0.208, 0.309, 0.25733333,
#      0.072],                                            # resnet-34, acc 0.92829
#
#     [0.197, 0.21333333, 0.216, 0.092, 0.16666667, 0.15133333,
#      0.178, 0.25033333, 0.165, 0.17566667, 0.184, 0.20566667,
#      0.24933333, 0.233, 0.217, 0.16733333, 0.06],       # resnet-50, acc 0.93020
#
#     [0.16433333, 0.227, 0.17366667,
#      0.05133333, 0.275, 0.16033333,
#      0.1943333, 0.22533333, 0.22033333,
#      0.16833333, 0.16366667, 0.34066667,
#      0.25333333, 0.13466667, 0.29566667,
#      0.209, 0.05533333],                                # resnet-152, 0.929617
#
#     [0.13166667, 0.22266667, 0.26733333,
#      0.062, 0.303, 0.18366667,
#      0.18966667, 0.305, 0.252,
#      0.29033333, 0.18766667, 0.15166667,
#      0.14066667, 0.04766667, 0.41466667,
#      0.26233333, 0.07333333],                           # densenet-121, acc 0.92821
#
#     [0.18533333, 0.18866667, 0.13533333,
#      0.03633333, 0.221, 0.17666667,
#      0.231, 0.23933333, 0.21966667,
#      0.169, 0.23333333, 0.21833333,
#      0.24033333, 0.112, 0.40233333,
#      0.31833333, 0.237],                                # densenet-161, 0.93108
#
#     [0.21266667, 0.18866667, 0.17733333,
#      0.07166667, 0.15366667, 0.14966667,
#      0.153, 0.17866667, 0.15966667,
#      0.15366667, 0.16133333, 0.126,
#      0.19066667, 0.09166667, 0.313,
#      0.25366667, 0.06266667],                           # densenet-169, 0.92856

    [0.167, 0.18633333, 0.206, 0.12966667, 0.26133333, 0.19666667,
      0.218, 0.20933333, 0.21133333, 0.24333333, 0.109, 0.23566667,
      0.168, 0.151, 0.13333333, 0.125, 0.05933333],

    [0.232, 0.185, 0.13233333, 0.13466667, 0.24066667, 0.25233333,
     0.18733333, 0.204, 0.139, 0.163, 0.11866667, 0.128,
     0.11266667, 0.125, 0.17, 0.13733333, 0.10833333],

    [0.165, 0.16166667, 0.201, 0.12366667, 0.24833333, 0.193, 0.201,
     0.178, 0.24766667, 0.25266667, 0.06533333, 0.19433333, 0.18433333,
     0.18466667, 0.223, 0.12666667, 0.04466667],

    [0.21766667, 0.21433333, 0.18966667, 0.12266667, 0.246, 0.169,
     0.18266667, 0.18933333, 0.199, 0.19633333, 0.17866667, 0.266,
     0.20366667, 0.18, 0.23566667, 0.197, 0.201],

    [0.11433333, 0.196, 0.24, 0.06566667, 0.192, 0.169,
     0.21933333, 0.23166667, 0.209, 0.235, 0.21933333, 0.12033333,
     0.13, 0.11033333, 0.39533333, 0.176, 0.09266667],

    [0.21633333, 0.21233333, 0.21833333, 0.13166667, 0.19066667, 0.187,
     0.21666667, 0.21466667, 0.18866667, 0.18966667, 0.18066667, 0.154,
     0.20533333, 0.26833333, 0.19666667, 0.201, 0.18833333],

    [0.22366667, 0.20366667, 0.17166667, 0.14633333, 0.20066667, 0.18966667,
     0.197, 0.20166667, 0.17233333, 0.21466667, 0.15566667, 0.197,
     0.16366667, 0.149,  0.25366667, 0.18333333, 0.26033333]
]


# threshold = [ 0.17733333, 0.213, 0.15766667, 0.049, 0.28733333, 0.18066667,
#               0.19666667, 0.212, 0.21566667, 0.17233333, 0.16466667, 0.274,
#               0.27833333, 0.10266667, 0.293, 0.241, 0.08366667]   # densenet-161 + resnet-152

# threshold = [ 0.142, 0.17, 0.122, 0.054, 0.188, 0.156, 0.228, 0.234, 0.142, 0.226,
#               0.188, 0.192, 0.192, 0.084, 0.242, 0.4, 0.126]      # densenet-161 + densenet-169 + resnet-152

# threshold = [0.136, 0.236, 0.144, 0.044, 0.226, 0.152, 0.214, 0.218, 0.162, 0.204,
#              0.194, 0.19, 0.234, 0.066, 0.236, 0.188, 0.106]        # densenet121+densenet161+densenet169+resnet152

transforms = [default, rotate90, rotate180, rotate270, verticalFlip, horizontalFlip]

models = [
            resnet18_planet,
            resnet34_planet,
            resnet50_planet,
            resnet152_planet,
            densenet121,
            densenet161,
            densenet169,
        ]


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
            name = 'full_data_{}.pth'.format(name)
            net = nn.DataParallel(net)
            net.load_state_dict(torch.load('models/{}'.format(name)))
            net.eval()
            # predict
            m_predictions = predict(net, dataloader)

            # save
            np.savetxt(X=m_predictions, fname='probs/{}_{}.txt'.format(t_name, name))
            probabilities[t_idx, m_idx] = m_predictions
    return probabilities


def find_best_threshold(labels, probabilities):
    threshold = np.zeros(17)
    acc = 0
    # iterate over transformations
    for t_idx in range(len(transforms)):
        # iterate over class labels
        t = np.ones(17) * 0.15
        selected_preds = probabilities[t_idx]
        selected_preds = np.mean(selected_preds, axis=0)
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
    print('AVG ACC,', acc/len(transforms))
    threshold = threshold / len(transforms)
    return threshold


def get_validation_loader():
    validation = KgForestDataset(
        split='train-40479',
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
    return valid_dataloader


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


def do_thresholding(names, models, labels):
    preds = np.empty((len(transforms), len(models), 40479, 17))
    print('filenames', names)
    for t_idx in range(len(transforms)):
        for m_idx in range(len(models)):
            preds[t_idx, m_idx] = np.loadtxt(names[t_idx + m_idx])
    t = find_best_threshold(labels=labels, probabilities=preds)
    return t


def get_files(excludes=None):
    file_names = glob.glob('probs/*.txt')
    file_names = [f for f in file_names if 'full_data' in f]
    names = []
    for filename in file_names:
        if not any([exclude in filename for exclude in excludes]):
            names.append(filename)
    return names


def predict_test_majority():
    """
    Majority voting method.
    """
    labels = np.empty((len(models), 61191, 17))
    for m_idx, model in enumerate(models):
        name = str(model).split()[1]
        print('predicting model {}'.format(name))
        net = nn.DataParallel(model().cuda())
        net.load_state_dict(torch.load('models/full_data_{}.pth'.format(name)))
        net.eval()
        preds = np.zeros((61191, 17))
        for t in transforms:
            test_dataloader.dataset.images = t(test_dataloader.dataset.images)
            print(t, name)
            pred = predict(net, dataloader=test_dataloader)
            preds = preds + pred
        # get predictions for the single model
        preds = preds/len(transforms)
        np.savetxt('submission_probs/full_data_{}.txt'.format(name), preds)
        # get labels
        preds = (preds > thresholds[m_idx]).astype(int)
        labels[m_idx] = preds

    # majority voting
    labels = labels.sum(axis=0)
    labels = (labels >= (len(models)//2)).astype(int)
    pred_csv(predictions=labels, name='majority_voting_ensembles_full_data')


def predict_test_averaging(t):
    preds = np.zeros((61191, 17))
    # imgs = test_dataloader.dataset.images.copy()
    # iterate over models
    for index, model in enumerate(models):
        name = str(model).split()[1]
        net = nn.DataParallel(model().cuda())
        net.load_state_dict(torch.load('models/{}.pth'.format(name)))
        net.eval()
        # iterate over transformations
        for transformation in transforms:
            # imgs = transformation(imgs)
            test_dataloader.dataset.images = transformation(test_dataloader.dataset.images)
            pred = predict(dataloader=test_dataloader, net=net)
            preds = preds + pred

    preds = preds/(len(models) * len(transforms))
    # preds = preds / len(models)
    pred_csv(predictions=preds, threshold=t, name='transforms-resnet152_densenet161_densent169-ensembels')


if __name__ == '__main__':
    # valid_dataloader = get_validation_loader()
    test_dataloader = get_test_dataloader()

    # save results to files
    # probabilities = probs(valid_dataloader)

    # get threshold
    # model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'densenet169']
    # for m in models:
    #     name = str(m).split()[1].strip('_planet')
    #     file_names = get_files([n for n in model_names if n != name])
    #     print('Model {}'.format(name))
    #     t = do_thresholding(file_names, labels=valid_dataloader.dataset.labels, models=[m])
    #     print(t)

    # average testing
    # predict_test_averaging(threshold)

    # majority voting
    predict_test_majority()
