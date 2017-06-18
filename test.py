import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from datasets import mean, std
from labels import *
from planet_models.densenet_planet import densenet121, densenet161, densenet169
from trainers.train_simplenet import evaluate
from torchvision.transforms import Compose, Normalize, Lambda
import numpy as np
from torch.autograd import Variable
import torch
from data.kgdataset import toTensor, KgForestDataset
from torch.utils.data.dataloader import DataLoader
from util import pred_csv


BEST_THRESHOLD = np.array([ 0.2071, 0.1986, 0.1296, 0.0363, 0.2355 , 0.1766, 0.2255, 0.257, 0.1922,
                            0.1462, 0.2676, 0.0931, 0.213, 0.1041, 0.2606, 0.2872, 0.151])


def test():
    net = nn.DataParallel(densenet161().cuda())
    net.load_state_dict(torch.load('models/densenet161.pth'))
    net.eval()

    dataset = KgForestDataset(split='test-61191', transform=Compose([
            Lambda(lambda x: toTensor(x)),
            Normalize(mean=mean, std=std)
            ]
        ), height=256, width=256, label_csv=None
    )

    test_loader = DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True)

    probs = np.empty((61191, 17))
    current = 0
    for batch_idx, (images, im_ids) in enumerate(test_loader):
        num = images.size(0)
        previous = current
        current = previous + num
        logits = net(Variable(images.cuda(), volatile=True))
        prob = F.sigmoid(logits)
        probs[previous:current, :] = prob.data.cpu().numpy()
        print('Batch Index ', batch_idx)

    pred_csv(probs, name='densenet161', threshold=BEST_THRESHOLD)
        # result = evaluate(model, images)
        # result = F.sigmoid(result)
        # result = result.data.cpu().numpy()
    #     probs.append(prob.data.cpu().numpy())
    #     for r, id in zip(result, im_ids):
    #         label = np.zeros_like(r)
    #         for i in range(17):
    #             label[i] = (r[i] > BEST_THRESHOLD[i]).astype(np.int)
    #         label = np.where(label == 1)[0]
    #         labels = [idx_to_label[index] for index in label]
    #         if len(r) == 0:
    #             print('id', id)
    #             print('r', r)
    #         imid_to_label[id] = sorted(labels)
    #     print('Batch Index {}'.format(batch_idx))
    # sample_submission = pd.read_csv('/media/jxu7/BACK-UP/Data/AmazonPlanet/sample_submission.csv')
    # for key, value in imid_to_label.items():
    #     sample_submission.loc[sample_submission['image_name'] == key,'tags'] = ' '.join(str(x) for x in value)
    # sample_submission.to_csv('submissions/%s.csv' % name, index=False)


if __name__ == '__main__':
    test()
