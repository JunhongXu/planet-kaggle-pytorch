from torchvision.transforms import *
import torch.nn as nn
from trainers.train_densenet import densenet121
from planet_models.resnet_planet import *
from planet_models.simplenet_v3 import SimpleNetV3
from planet_models.simplenet_v2 import *
from datasets import *
from trainers.train_simplenet import evaluate
import torch.nn.functional as F
from util import f2_score


def optimize_threshold(resolution=1000):
    """
    This function takes the validation set and find the best threshold for each class.
    """
    def get_labels(prediction, t):
        pred = np.zeros_like(prediction)
        for i in range(0, 17):
            pred[:, i] = (prediction[:, i] > t[i]).astype(np.int)
        return pred

    large_net = nn.DataParallel(densenet121().cuda())
    large_net.load_state_dict(torch.load('../models/densenet121.pth'))
    large_net.eval()
    # simple_v2 = nn.DataParallel(SimpleNetV3().cuda())
    # simple_v2.load_state_dict(torch.load('../models/simplenet_v3.1.pth'))
    # simple_v2.eval()

    resnet_data = validation_jpg_loader(512, transform=Compose([
        Scale(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std)
    ]))

    # simplenet_data = validation_jpg_loader(
    #     512, transform=Compose(
    #         [
    #             Scale(72),
    #             RandomHorizontalFlip(),
    #             ToTensor(),
    #             Normalize(mean, std)
    #         ]
    #     )
    # )
    num_class = 17
    pred = []
    targets = []
    # predict
    # for batch_index, ((resnet_img, resnet_target), (simplenet_img, simplenet_target)) \
    #     in enumerate(zip(resnet_data, simplenet_data)):
    for batch_index, (simplenet_img, simplenet_target) in enumerate(resnet_data):
        resnet_output = evaluate(large_net, simplenet_img)
        # resnet_output = F.sigmoid(resnet_output)

        # simplenet_output = evaluate(simple_v2, simplenet_img)
        # simplenet_output = F.sigmoid(simplenet_output)

        # output = F.sigmoid((simplenet_output + resnet_output)/2)
        output = F.sigmoid(resnet_output)
        pred.append(output.data.cpu().numpy())
        targets.append(simplenet_target.cpu().numpy())

    pred = np.vstack(pred)
    targets = np.vstack(targets)
    threshold = [0.2] * 17
    # optimize
    for i in range(num_class):
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


if __name__ == '__main__':
    threshold = np.zeros(17)
    for i in range(0,10):
        threshold = threshold + np.array(optimize_threshold(100))
    threshold = threshold / 10
    print(threshold)
