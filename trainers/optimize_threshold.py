from planet_models.densenet_planet import *
from planet_models.resnet_planet import *
from torch.autograd import Variable
from datasets import *
import torch.nn.functional as F
from util import f2_score
from data.kgdataset import KgForestDataset, randomTranspose, randomFlip, toTensor


def save_results(model, dataset, name):
    """This function saves the probability to name.txt file"""
    result = []
    for image, target, _ in dataset:
        image = Variable(image.cuda(), volatile=True)
        logit = model(image)
        probs = F.sigmoid(logit)
        result.append(probs.data.cpu().numpy())

    # concatenate the probabilities
    result = np.concatenate(result)
    # save the probabilities into model.txt file
    np.savetxt(fname='probs/{}.txt'.format(name), X=result)


def optimize_threshold(models, datasets, resolution=1000):
    """
    This function takes the validation set and find the best threshold for each class.
    """
    pred = []
    targets = []
    # predict
    # for index, data in enumerate(zip(*datasets)):
    for batch_index, (image, target, _) in enumerate(datasets[0]):
        image = Variable(image.cuda(), volatile=True)
        output = F.sigmoid(models[0](image))
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
    model1 = nn.DataParallel(densenet121(pretrained=True).cuda())
    model1.load_state_dict(torch.load('../models/full_data_densenet121.pth'))
    model1.cuda().eval()
    validation = KgForestDataset(
        split='valid-8000',
        transform=Compose(
            [
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


