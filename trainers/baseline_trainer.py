import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision.transforms import Lambda, Compose, Normalize
from planet_models.densenet_planet import densenet169, densenet121, densenet161
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import mean, std
from data.kgdataset import KgForestDataset, randomShiftScaleRotate, randomFlip, randomTranspose, toTensor
from util import Logger, evaluate, multi_criterion, multi_f_measure, lr_schedule, get_learning_rate
import numpy as np
import torch
import time

"""
A baseline trainer trains the models as followed:
1. ResNet: 18, 34, 50, and 152 (from scratch)
2. DenseNet: 169, 161, and 121 (from scratch)

-------parameters---------
    epochs: 80

    batch size: 128, 128, 128, 64, 64, 64, 64

    use SGD+0.9momentum w/o nestrov

    weight decay: 5e-4

    learning rate: 00-10 epoch: 0.1
                   10-25 epoch: 0.01
                   25-35 epoch: 0.005
                   35-40 epoch: 0.001
                   40-80 epoch: 0.0001

    transformations: Rotate, VerticalFlip, HorizontalFlip, RandomCrop
"""


models = [
        resnet18_planet, resnet34_planet, resnet50_planet,
        densenet121, densenet169, densenet161,
          ]
batch_size = [128, 128, 128, 64,
                64, 64, 64]


def get_dataloader(batch_size):
    train_data = KgForestDataset(
        split='train-37479',
        transform=Compose(
            [
                Lambda(lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=6, scale_limit=6, rotate_limit=45)),
                Lambda(lambda x: randomFlip(x)),
                Lambda(lambda x: randomTranspose(x)),
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )
    train_data_loader = DataLoader(batch_size=batch_size, dataset=train_data, shuffle=True)

    validation = KgForestDataset(
        split='validation-3000',
        transform=Compose(
            [
                # Lambda(lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=6, scale_limit=6, rotate_limit=45)),
                # Lambda(lambda x: randomFlip(x)),
                #  Lambda(lambda x: randomTranspose(x)),
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )

    valid_dataloader = DataLoader(dataset=validation, shuffle=False, batch_size=batch_size)
    return train_data_loader, valid_dataloader


def train_baselines():

    train_data, val_data = get_dataloader(96)

    for model, batch in zip(models, batch_size):
        name = str(model).split()[1]
        print('*****Start Training {} with batch size {}******'.format(name, batch))
        print(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | total_train_loss\n')
        logger = Logger('../log/{}'.format(name), name)

        net = model()
        net = nn.DataParallel(net.cuda())

        train_data.batch_size = batch
        val_data.batch_size = batch

        num_epoches = 100  #100
        print_every_iter = 20
        epoch_test = 1

        # optimizer
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)

        smooth_loss = 0.0
        train_loss = np.nan
        train_acc = np.nan
        # test_loss = np.nan
        best_test_loss = np.inf
        # test_acc = np.nan
        t = time.time()

        for epoch in range(num_epoches):  # loop over the dataset multiple times
            # train loss averaged every epoch
            total_epoch_loss = 0.0

            # lr_schedule(epoch, optimizer)

            rate = get_learning_rate(optimizer)[0]  # check

            sum_smooth_loss = 0.0
            total_sum = 0
            sum = 0
            net.cuda().train()

            num_its = len(train_data)
            for it, (images, labels, indices) in enumerate(train_data, 0):

                logits = net(Variable(images.cuda()))
                probs = F.sigmoid(logits)
                loss = multi_criterion(logits, labels.cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # additional metrics
                sum_smooth_loss += loss.data[0]
                total_epoch_loss += loss.data[0]
                sum += 1
                total_sum += 1

                # print statistics
                if it % print_every_iter == print_every_iter-1:
                    smooth_loss = sum_smooth_loss/sum
                    sum_smooth_loss = 0.0
                    sum = 0

                    train_acc = multi_f_measure(probs.data, labels.cuda())
                    train_loss = loss.data[0]
                    print('\r{}   {}    {}   |  {}  | {}  {} | ... '.
                          format(epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),
                          end='', flush=True)

            total_epoch_loss = total_epoch_loss / total_sum
            if epoch % epoch_test == epoch_test-1 or epoch == num_epoches-1:
                net.cuda().eval()
                test_loss, test_acc = evaluate(net, val_data)
                print('\r', end='', flush=True)
                print('{}   {}    {}   |  {}  | {}  {} | {}  {} | {}'.
                      format(epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss, test_acc,
                             total_epoch_loss))

                # save if the current loss is better
                if test_loss < best_test_loss:
                    torch.save(net.state_dict(), '../models/{}.pth'.format(name))
                    best_test_loss = test_loss

            logger.add_record('train_loss', total_epoch_loss)
            logger.add_record('evaluation_loss', test_loss)
            logger.add_record('f2_score', test_acc)

            logger.save()
            logger.save_plot()
            logger.save_time(start_time=t, end_time=time.time())


if __name__ == '__main__':
    train_baselines()
