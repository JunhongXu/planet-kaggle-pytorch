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
# threshold = [0.23166666666666666, 0.19599999999999998, 0.18533333333333335,
#              0.08033333333333334, 0.20199999999999999, 0.16866666666666666,
#              0.20533333333333334, 0.27366666666666667, 0.2193333333333333,
#              0.21299999999999999, 0.15666666666666665, 0.096666666666666679,
 #            0.21933333333333335, 0.058666666666666673, 0.19033333333333333,#           0.25866666666666666, 0.057999999999999996]

# threshold = [ 0.18533333,  0.18866667, 0.13533333, 0.03633333, 0.221, 0.17666667,
  # 0.231, 0.23933333, 0.21966667, 0.169, 0.23333333, 0.21833333,
 #  0. 24033333, 0.112, 0.40233333, 0.31833333, 0.237]

# threshold = [ 0.17733333, 0.213, 0.15766667, 0.049, 0.28733333, 0.18066667,
 #  0.19666667, 0.212, 0.21566667, 0.17233333, 0.16466667, 0.274,
 #  0.27833333, 0.10266667, 0.293, 0.241, 0.08366667]
threshold = [ 0.142, 0.17, 0.122, 0.054, 0.188, 0.156, 0.228, 0.234, 0.142, 0.226,
  0.188, 0.192, 0.192, 0.084, 0.242, 0.4, 0.126]

transforms = [default, rotate90, rotate180, rotate270, verticalFlip, horizontalFlip]
models = [# resnet18_planet, resnet34_planet, resnet50_planet, densenet121, densenet161, densenet169
           # resnet152_planet
    densenet121,
    densenet169,
            densenet161,
            resnet152_planet
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
            net = nn.DataParallel(net)
            net.load_state_dict(torch.load('models/{}.pth'.format(name)))
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
            print('Transform index {}, score {}, threshold {}, label {}'.format(t_idx, best_score, best_thresh, i))
        print('Transform index {}, threshold {}, score {}'.format(t_idx, t, best_score))
        threshold = threshold + t
        acc += best_score
    print('AVG ACC,', acc/len(transforms))
    threshold = threshold / len(transforms)
    return threshold


def get_validation_loader():
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


def do_thresholding(names, labels):
    preds = np.empty((len(transforms), len(models), 3000, 17))
    print('filenames', names)
    for t_idx in range(len(transforms)):
        for m_idx in range(len(models)):
            preds[t_idx, m_idx] = np.loadtxt(names[t_idx + m_idx])
    t = find_best_threshold(labels=labels, probabilities=preds)
    return t


def get_files(excludes=['resnet18']):
    file_names = glob.glob('probs/*.txt')
    names = []
    for filename in file_names:
        if not any([exclude in filename for exclude in excludes]):
            names.append(filename)
    return names


def predict_test(t):
    preds = np.zeros((61191, 17))
    for index, model in enumerate(models):
        name = str(model).split()[1]
        net = nn.DataParallel(model().cuda())
        net.eval()
        net.load_state_dict(torch.load('models/{}.pth'.format(name)))
        pred = predict(dataloader=test_dataloader, net=net)
        preds = preds + pred

    preds = preds/len(models)
    pred_csv(predictions=preds, threshold=t, name='resnet152_densenet161_densent169_densent121-ensembels')


if __name__ == '__main__':
    # valid_dataloader = get_validation_loader()
    test_dataloader = get_test_dataloader()

    # save results to files
    # probabilities = probs(valid_dataloader)

    # get threshold
    # file_names = get_files(['resnet18', 'resnet50', 'resnet34'])
    # t = do_thresholding(file_names, valid_dataloader.dataset.labels)
   #  print(t)

    # testing
    predict_test(threshold)
