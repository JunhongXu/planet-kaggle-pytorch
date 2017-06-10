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

#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
#f  = fbeta_score(labels, probs, beta=2, average='samples')
def multi_f_measure( probs, labels, threshold=0.235, beta=2 ):
    batch_size = probs.size()[0]
    SMALL = 1e-12
    #weather
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f


def evaluate(net, test_loader):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits = net(Variable(images.cuda(),volatile=True))
        probs = F.sigmoid(logits)
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc


def get_dataloader(batch_size):
    train_data = KgForestDataset(
        split='train-32479',
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
        split='valid-8000',
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


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


# def evaluate_train(model, val_data, criterion, b):
#     # evaluating
#     val_loss = 0.0
#     model.eval()
#     preds = []
#     targets = []
#     # num_images = len(val_data.dataset)
#     num_img = 0
#     for batch_index, (val_x, val_y, index) in enumerate(val_data):
#         if torch.cuda.is_available():
#             val_y = val_y.cuda()
#         val_y = Variable(val_y, volatile=True)
#         val_output = evaluate(model, val_x)
#         val_loss += criterion(val_output, val_y)*b
#         val_output = F.sigmoid(val_output)
#         binary_y = val_output.data.cpu().numpy()
#         binary_y = (binary_y > 0.2).astype(np.int32)
#         preds.append(binary_y)
#         targets.append(val_y.data.cpu().numpy())
#         num_img += val_x.size(0)
#     targets = np.concatenate(targets)
#     preds = np.concatenate(preds)
#     f2_scores = f2_score(targets, preds)
#     val_loss = val_loss.data[0]/num_img
#     return val_loss, f2_scores


def train_baselines(epoch):
    # transformations = Compose(
    #     [
    #         Lambda(lambda x: randomShiftScaleRotate(x)),
    #         Lambda(lambda x: randomFlip(x)),
    #         RandomTranspose(),
    #         Lambda(lambda x: toTensor(x)),
    #         # RandomRotate(),
    #         # RandomCrop(224),
    #         # Normalize(mean=mean, std=std)
    #      ]
    # )

    # criterion = nn.MultiLabelSoftMarginLoss()
    # train_data = train_jpg_loader(64, transform=transformations)
    # val_data = validation_jpg_loader(64, transform=Compose(
    #         [
    #             # Scale(224),
    #             Lambda(lambda x: toTensor(x)),
    #             # Normalize(mean=mean, std=std)
    #         ]
    #     ))

    train_data, val_data = get_dataloader(96)

    for model, batch in zip(models, batch_size):
        net = model()
        net = nn.DataParallel(net.cuda())


        ## optimiser ----------------------------------
        num_epoches = 50  #100
        it_print    = 20
        epoch_test  = 1
        epoch_save  = 5


        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

        ## start training here! ###

        smooth_loss = 0.0
        train_loss  = np.nan
        train_acc   = np.nan
        test_loss   = np.nan
        test_acc    = np.nan
        time = 0

        for epoch in range(num_epoches):  # loop over the dataset multiple times
            #print ('epoch=%d'%epoch)

            if 1:
                lr = 0.1 # schduler here ---------------------------
                if epoch>=10: lr=0.010
                if epoch>=25: lr=0.005
                if epoch>=35: lr=0.001
                if epoch>=40: lr=0.0001
                if epoch> 42: break

                adjust_learning_rate(optimizer, lr)

            rate =  get_learning_rate(optimizer)[0] #check


            sum_smooth_loss = 0.0
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

                #additional metrics
                sum_smooth_loss += loss.data[0]
                sum += 1

                # print statistics
                if it % it_print == it_print-1:
                    smooth_loss = sum_smooth_loss/sum
                    sum_smooth_loss = 0.0
                    sum = 0

                    train_acc = multi_f_measure(probs.data, labels.cuda())
                    train_loss = loss.data[0]

                    print('\r%5.1f   %5d    %0.4f   |  %0.3f  | %0.3f  %5.3f | ... ' % \
                            (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                            end='',flush=True)



            if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:

                net.cuda().eval()
                test_loss,test_acc = evaluate(net, val_data)

                print('\r',end='',flush=True)

            if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
                torch.save(net, 'snap/%03d.torch'%(epoch+1))
        # name = str(model).split()[1]
        # print('[!]Training %s' % name)
        # print('[!]Batch size %s' % batch)
        # logger = Logger(name=name, save_dir='../log/%s' % name)
        # # model = nn.DataParallel(model().cuda())
        # model = model().cuda()
        # optimizer = optim.SGD(momentum=0.9, lr=0.1, params=model.parameters(), weight_decay=1e-4)
        #
        # train_data.batch_size = batch
        # val_data.batch_size = batch
        #
        # # start training
        # best_loss = np.inf
        # patience = 0
        # start_time = time.time()
        # for i in range(epoch):
        #     # training
        #     training_loss = 0.0
        #     # adjust learning rate
        #     lr_schedule(epoch, optimizer)
        #     num_img = 0
        #     for batch_index, (target_x, target_y, index) in enumerate(train_data):
        #         if torch.cuda.is_available():
        #             target_x, target_y = target_x.cuda(), target_y.cuda()
        #         num_img += target_x.size(0)
        #         model.train()
        #         target_x, target_y = Variable(target_x), Variable(target_y)
        #         output = model(target_x)
        #         loss = criterion(output, target_y)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         training_loss += loss.data[0] * batch
        #         if batch_index % 50 == 0:
        #             print('Training loss is {}'.format(loss.data[0]))
        #     print('Finished epoch {}'.format(i))
        #     training_loss /= num_img
        #
        #     # evaluating
        #     val_loss, f2_scores = evaluate_train(model, val_data, criterion, batch)
        #
        #     if best_loss > val_loss:
        #         print('Saving model...')
        #         best_loss = val_loss
        #         torch.save(model.state_dict(), '../models/{}.pth'.format(name))
        #         patience = 0
        #     else:
        #         patience += 1
        #         print('Patience: {}'.format(patience))
        #         print('Best loss {}, previous loss {}'.format(best_loss, val_loss))
        #
        #     print('Evaluation loss is {}, Training loss is {}'.format(val_loss, training_loss))
        #     print('F2 Score is %s' % (f2_scores))
        #
        #     logger.add_record('train_loss', training_loss)
        #     logger.add_record('evaluation_loss', val_loss)
        #     logger.add_record('f2_score', f2_scores)
        #
        #     # save for every epoch
        #     logger.save()
        #     logger.save_plot()
        #
        # logger.save_time(start_time, time.time())
if __name__ == '__main__':
    train_baselines(150)
