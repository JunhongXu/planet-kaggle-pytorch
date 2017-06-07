import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import *
from planet_models.resnext import resnext_29
from datasets import input_transform
from datasets import test_jpg_loader, mean, std
from labels import *
from planet_models.simplenet import MultiLabelCNN
from planet_models.simplenet_v2 import SimpleNetV2

from planet_models.densenet_planet import densenet121, densenet169
from planet_models.simplenet_v3 import SimpleNetV3
from planet_models.resnet_planet import *
from trainers.train_simplenet import evaluate
from util import BEST_THRESHOLD


MODEL='models/pretrained_densenet169_wd_1e-4.pth'


def test(model_dir, transform):
    name = model_dir.split('/')[-1][:-4]
    test_loader = test_jpg_loader(512, transform=Compose(
        [
            Scale(224),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))

    # if 'resnet' in model_dir:
    #     model = nn.DataParallel(resnet14_planet())
    # elif 'resnext' in model_dir:
    #     model = nn.DataParallel(resnext_29())
    # else:
    #     model = nn.DataParallel(SimpleNetV3())
    model = nn.DataParallel(densenet169())
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    imid_to_label = {}
    if torch.cuda.is_available():
        model.cuda()
    for batch_idx, (images, im_ids) in enumerate(test_loader):
        result = evaluate(model, images)
        result = F.sigmoid(result)
        result = result.data.cpu().numpy()
        for r, id in zip(result, im_ids):
            label = np.zeros_like(r)
            for i in range(17):
                label[i] = (r[i] > BEST_THRESHOLD[i]).astype(np.int)
            label = np.where(label == 1)[0]
            labels = [idx_to_label[index] for index in label]
            if len(r) == 0:
                print('id', id)
                print('r', r)
            imid_to_label[id] = sorted(labels)
        print('Batch Index {}'.format(batch_idx))
    sample_submission = pd.read_csv('/media/jxu7/BACK-UP/Data/AmazonPlanet/sample_submission.csv')
    for key, value in imid_to_label.items():
        sample_submission.loc[sample_submission['image_name'] == key,'tags'] = ' '.join(str(x) for x in value)
    sample_submission.to_csv('submissions/%s.csv' % name, index=False)


if __name__ == '__main__':
    if 'resnet' in MODEL:
        transform = input_transform(227)
    else:
        transform = ToTensor()
    test(MODEL, transform=transform)
