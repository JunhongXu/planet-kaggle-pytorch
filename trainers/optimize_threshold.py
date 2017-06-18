from torchvision.transforms import *
import torch.nn as nn
from trainers.train_densenet import densenet121
from planet_models.densenet_planet import *
from planet_models.resnet_planet import *
from torch.autograd import Variable
from planet_models.simplenet_v2 import *
from datasets import *
from trainers.train_simplenet import evaluate
import torch.nn.functional as F
from util import f2_score
from data.kgdataset import KgForestDataset, randomTranspose, randomFlip, toTensor
from trainers.baseline_trainer import get_dataloader

def optimize_threshold_single(resolution=1000):
    """
    This function takes the validation set and find the best threshold for each class.
    """
    def get_labels(prediction, t):
        pred = np.zeros_like(prediction)
        for i in range(0, 17):
            pred[:, i] = (prediction[:, i] > t[i]).astype(np.int)
        return pred

    net = nn.DataParallel(densenet169(pretrained=False).cuda())
    net.load_state_dict(torch.load('../models/pretrained_densenet169_wd_1e-4.pth'))
    net.eval()

    resnet_data = validation_jpg_loader(512, transform=Compose([
        Scale(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std)
    ]))
    pred = []
    targets = []
    # predict
    for batch_index, (img, target) in enumerate(resnet_data):
        output = evaluate(net, img)
        output = F.sigmoid(output)
        pred.append(output.data.cpu().numpy())
        targets.append(target.cpu().numpy())

    pred = np.vstack(pred)
    targets = np.vstack(targets)
    threshold = [0.2] * 17
    # optimize
    for i in range(17):
        best_thresh = 0.0
        best_score = 0.0
        for r in range(resolution):
            r /= resolution
            threshold[i] = r
            labels = get_labels(pred, threshold)
            score = f2_score(targets, labels)
            if score > best_score:
                best_thresh = r
                best_score = score
        threshold[i] = best_thresh
        print(i, best_score, best_thresh)
    return threshold


def optimize_threshold(models, datasets, resolution=1000):
    """
    This function takes the validation set and find the best threshold for each class.
    """
    def get_labels(prediction, t):
        pred = np.zeros_like(prediction)
        for i in range(0, 17):
            pred[:, i] = (prediction[:, i] > t[i]).astype(np.int)
        return pred

    pred = []
    targets = []
    # predict
    # for index, data in enumerate(zip(*datasets)):
    for batch_index, (image, target, _) in enumerate(datasets[0]):
        # output += F.sigmoid(evaluate(models[index], image))
        image = Variable(image.cuda(), volatile=True)
        output = F.sigmoid(models[0](image))
        # print(output[0].data)
        # print(target[0])
        # output = output.data.cpu().numpy()
        # target = target.cpu().numpy()
        # print(f2_score(target, (output > 0.2).astype(int)))
        # print(target)
        # output = output/len(models)
        pred.append(output.data.cpu().numpy())
        targets.append(target.cpu().numpy())

    pred = np.vstack(pred)
    targets = np.vstack(targets)
    threshold = [0.15] * 17
    # optimize
    for i in range(17):
        best_thresh = 0.0
        best_score = 0.0
        for r in range(resolution):
            r /= resolution
            threshold[i] = r
            labels = (pred > threshold).astype(int)
            score = f2_score(targets, labels)
            if score > best_score:
                best_thresh = r
                best_score = score
        threshold[i] = best_thresh
        print(i, best_score, best_thresh)
    return threshold


if __name__ == '__main__':
    model1 = nn.DataParallel(densenet161(pretrained=True).cuda())
    model1.load_state_dict(torch.load('../models/densenet161.pth'))
    model1.cuda().eval()
    validation = KgForestDataset(
        split='validation-3000',
        transform=Compose(
            [
                # Lambda(lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=6, scale_limit=6, rotate_limit=45)),
                Lambda(lambda x: randomFlip(x)),
                Lambda(lambda x: randomTranspose(x)),
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )

    valid_dataloader = DataLoader(dataset=validation, shuffle=False, batch_size=512)

    threshold = np.zeros(17)
    for i in range(10):
        threshold += optimize_threshold([model1], [valid_dataloader])
    print(threshold/10)


