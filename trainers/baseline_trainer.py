import torch.nn as nn
from torch.nn import functional as F
from torch import optim
# from trainers.train_simplenet import evaluate
from torchvision.transforms import Lambda, Compose, Normalize
from planet_models.densenet_planet import densenet169, densenet121, densenet161
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from datasets import RandomTranspose, randomFlip, \
#     train_jpg_loader, validation_jpg_loader, mean, std, toTensor, randomShiftScaleRotate, train_jpg_loader_all

from data.kgdataset import KgForestDataset, randomShiftScaleRotate, randomFlip, randomTranspose, toTensor
from util import Logger, f2_score
import numpy as np
import torch
import time

"""
A baseline trainer trains the models as followed:
1. ResNet: 18, 34, 50, and 152 (from scratch)
2. DenseNet: 169, 161, and 121 (from scratch)

-------parameters---------
    epochs: 80

    batch size: 96, 96, 96, 60, 60, 60, 60

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
        resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet,
        densenet121, densenet169, densenet161,
          ]
batch_size = [96, 96, 96, 60, 60, 60, 60]


# loss ----------------------------------------
def multi_criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss


def multi_f_measure(probs, labels, threshold=0.235, beta=2):
    batch_size = probs.size()[0]
    SMALL = 1e-12
    l = labels
    p = (probs > threshold).float()
    num_pos = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp = torch.sum(l*p,1)
    precise = tp/(num_pos+ SMALL)
    recall = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f = fs.sum()/batch_size
    return f


def evaluate(net, test_loader):

    test_num = 0
    test_loss = 0
    test_acc = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits = net(Variable(images.cuda(), volatile=True))
        probs = F.sigmoid(logits)
        loss = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc


def get_dataloader(batch_size):
    train_data = KgForestDataset(
        split='train-37479',
        transform=Compose(
            [
                Lambda(lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=6, scale_limit=6, rotate_limit=45)),
                Lambda(lambda x: randomFlip(x)),
                Lambda(lambda x: randomTranspose(x)),
                Lambda(lambda x: toTensor(x))
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
                Lambda(lambda x: toTensor(x))
            ]
        ),
        height=256,
        width=256
    )

    valid_dataloader = DataLoader(dataset=validation, shuffle=False, batch_size=batch_size)
    return train_data_loader, valid_dataloader


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


def lr_schedule(epoch, optimizer):
    if 0 <= epoch < 10:
        lr = 1e-1
    elif 10 <= epoch < 25:
        lr = 0.01
    elif 25 <= epoch < 35:
        lr = 0.005
    elif 35 <= epoch < 40:
        lr = 0.001
    else:
        lr = 0.0001

    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


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

        num_epoches = 50  #100
        print_every_iter = 20
        epoch_test = 1

        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

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

            lr_schedule(epoch, optimizer)

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
                    torch.save(net, '../models/{}.pth'.format(name))
                    best_test_loss = test_loss

            logger.add_record('train_loss', total_epoch_loss)
            logger.add_record('evaluation_loss', test_loss)
            logger.add_record('f2_score', test_acc)

            logger.save()
            logger.save_plot()
            logger.save_time(start_time=t, end_time=time.time())


if __name__ == '__main__':
    train_baselines()
