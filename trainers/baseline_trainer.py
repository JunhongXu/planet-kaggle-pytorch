import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from trainers.train_simplenet import evaluate
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Compose, Scale, ToTensor, Normalize
from planet_models.densenet_planet import densenet169, densenet121
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet
from torch.autograd import Variable
from datasets import RandomRotate, RandomVerticalFlip, RandomTranspose, train_jpg_loader, validation_jpg_loader, mean, std
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

    batch size: 128, 128, 128, 72, 72, 72, 72

    use SGD+0.9momentum w/o nestrov

    weight decay: 5e-4

    learning rate: 00-10 epoch: 0.1
                   10-25 epoch: 0.01
                   25-35 epoch: 0.005
                   35-40 epoch: 0.001
                   40-80 epoch: 0.0001

    transformations: Rotate, VerticalFlip, HorizontalFlip, RandomCrop
"""


models = [resnet18_planet, resnet34_planet, resnet50_planet, resnet152_planet, densenet169, densenet121]
batch_size = [128, 128, 128, 72, 72, 72, 72]


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


def evaluate_train(model, val_data, criterion):
    # evaluating
    val_loss = 0.0
    model.eval()
    preds = []
    targets = []
    for batch_index, (val_x, val_y) in enumerate(val_data):
        if torch.cuda.is_available():
            val_y = val_y.cuda()
        val_y = Variable(val_y, volatile=True)
        val_output = evaluate(model, val_x)
        val_loss += criterion(val_output, val_y)
        val_output = F.sigmoid(val_output)
        binary_y = val_output.data.cpu().numpy()
        binary_y[binary_y > 0.2] = 1
        binary_y[binary_y <= 0.2] = 0
        preds.append(binary_y)
        targets.append(val_y.data.cpu().numpy())
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    f2_scores = f2_score(targets, preds)
    val_loss = val_loss.data[0]/batch_index
    return val_loss, f2_scores


def train_baselines(epoch):
    transformations = Compose(
        [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomTranspose(),
            RandomRotate(),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=mean, std=std)
         ]
    )

    criterion = nn.MultiLabelSoftMarginLoss()

    for model, batch in zip(models, batch_size):
        name = str(model).split()[1]
        print('[!]Training %s' % name)
        print('[!]Batch size %s' % batch)
        logger = Logger(name=name, save_dir='../log/%s' % name)
        model = nn.DataParallel(model().cuda())
        optimizer = optim.SGD(momentum=0.9, lr=0.1, params=model.parameters(), weight_decay=5e-4)

        train_data = train_jpg_loader(batch, transform=transformations)
        val_data = validation_jpg_loader(batch, transform=Compose(
            [
                Scale(224),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ]
        ))


        # start training
        best_loss = np.inf
        patience = 0
        start_time = time.time()
        for i in range(epoch):
            # training
            training_loss = 0.0
            # adjust learning rate
            lr_schedule(epoch, optimizer)
            for batch_index, (target_x, target_y) in enumerate(train_data):
                if torch.cuda.is_available():
                    target_x, target_y = target_x.cuda(), target_y.cuda()
                model.train()
                target_x, target_y = Variable(target_x), Variable(target_y)
                optimizer.zero_grad()
                output = model(target_x)
                loss = criterion(output, target_y)
                training_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                if batch_index % 50 == 0:
                    print('Training loss is {}'.format(loss.data[0]))
            print('Finished epoch {}'.format(i))
            training_loss /= batch_index

            # evaluating
            val_loss, f2_scores = evaluate_train(model, val_data, criterion)

            if best_loss > val_loss:
                print('Saving model...')
                best_loss = val_loss
                torch.save(model.state_dict(), '../models/{}.pth'.format(name))
                patience = 0
            else:
                patience += 1
                print('Patience: {}'.format(patience))
                print('Best loss {}, previous loss {}'.format(best_loss, val_loss))

            print('Evaluation loss is {}, Training loss is {}'.format(val_loss, training_loss))
            print('F2 Score is %s' % (f2_scores))

            logger.add_record('train_loss', training_loss)
            logger.add_record('evaluation_loss', val_loss)
            logger.add_record('f2_score', f2_scores)

            # save for every epoch
            logger.save()
            logger.save_plot()

        logger.save_time(start_time, time.time())
if __name__ == '__main__':
    train_baselines(80)
