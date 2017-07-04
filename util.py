import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import fbeta_score
from torch.nn import functional as F
from matplotlib import pyplot as plt
import pandas as pds
from datasets import *
import torch
import os
from data.kgdataset import CLASS_NAMES, KAGGLE_DATA_DIR
import pandas as pd


def name_idx():
    return {name: idx for idx, name in enumerate(CLASS_NAMES)}


def idx_name():
    return {idx: name for idx, name in enumerate(CLASS_NAMES)}


def predict(net, dataloader):
    num = dataloader.dataset.num
    probs = np.empty((num, 17))
    current = 0
    for batch_idx, (images, im_ids, _) in enumerate(dataloader):
        num = images.size(0)
        previous = current
        current = previous + num
        logits = net(Variable(images.cuda(), volatile=True))
        prob = F.sigmoid(logits)
        probs[previous:current, :] = prob.data.cpu().numpy()
        print('Batch Index ', batch_idx)
    return probs


def pred_csv(predictions, name, threshold=None):
    """
    predictions: numpy array of predicted probabilities
    """
    csv_name = os.path.join(KAGGLE_DATA_DIR, 'sample_submission.csv')
    submission = pd.read_csv(csv_name)
    for i, pred in enumerate(predictions):
        if threshold is not None:
            labels = (pred > threshold).astype(int)
        else:
            labels = pred
        labels = np.where(labels == 1)[0]
        labels = ' '.join(idx_name()[index] for index in labels)
        submission['tags'][i] = labels
        print('Index ', i)
    submission.to_csv(os.path.join('submissions', '{}.csv'.format(name)), index=False)


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


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            lr +=[param_group['lr']]
    return lr


def lr_schedule(epoch, optimizer, base_lr=0.1, pretrained=False):
    if pretrained:
        if 0 <= epoch < 10:
            lr = base_lr
        elif 10 <= epoch < 25:
            lr = base_lr * 0.5
        elif 25 <= epoch < 40:
            lr = base_lr * 0.1
        else:
            lr = base_lr * 0.01
    else:
        if 0 <= epoch < 10:
            lr = 1e-1
        elif 10 <= epoch < 25:
            lr = 5e-2
        elif 25 <= epoch < 40:
            lr = 1e-2
        else:
            lr = 1e-3

    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


def split_train_validation(num_val=3000):
    """
    Save train image names and validation image names to csv files
    """
    train_image_idx = np.sort(np.random.choice(40479, 40479-num_val, replace=False))
    all_idx = np.arange(40479)
    validation_image_idx = np.zeros(num_val, dtype=np.int32)
    val_idx = 0
    train_idx = 0
    for i in all_idx:
        if i not in train_image_idx:
            validation_image_idx[val_idx] = i
            val_idx += 1
        else:
            train_idx += 1
    # save train
    train = []
    for name in train_image_idx:
        train.append('train-<ext>/train_%s.<ext>' % name)

    eval = []
    for name in validation_image_idx:
        eval.append('train-<ext>/train_%s.<ext>' % name)

    df = pds.DataFrame(train)
    df.to_csv('dataset/train-%s' % (40479 - num_val), index=False, header=False)

    df = pds.DataFrame(eval)
    df.to_csv('dataset/validation-%s' % num_val, index=False, header=False)


def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


class Logger(object):
    def __init__(self, save_dir, name):
        self.save_dir = save_dir
        self.name = name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_dict = {'train_loss': [], "evaluation_loss": [], 'f2_score': []}

    def add_record(self, key, value):
        self.save_dict[key].append(value)

    def save(self):

        df = pd.DataFrame.from_dict(self.save_dict)
        df.to_csv(os.path.join(self.save_dir, '%s.csv' % self.name), header=True, index=False)

    def save_plot(self):
        train_loss = self.save_dict['train_loss']
        eval_loss = self.save_dict['evaluation_loss']
        f2_scores = self.save_dict['f2_score']
        plt.figure()
        plt.plot(np.arange(len(train_loss)), train_loss, color='red', label='train_loss')
        plt.plot(np.arange(len(eval_loss)), eval_loss, color='blue', label='eval_loss')
        plt.legend(loc='best')

        plt.savefig(os.path.join(self.save_dir, 'loss.jpg'))

        plt.figure()
        plt.plot(np.arange(len(f2_scores)), f2_scores)
        plt.savefig(os.path.join(self.save_dir, 'f2_score.jpg'))

        plt.close('all')

    def save_time(self, start_time, end_time):
        with open(os.path.join(self.save_dir, 'time.txt'), 'w') as f:
            f.write('start time, end time, duration\n')
            f.write('{}, {}, {}'.format(start_time, end_time, (end_time - start_time)/60))


if __name__ == '__main__':

    files = ['probs/densenet121.txt', 'probs/densenet161.txt', 'probs/densenet169.txt', 'probs/resnet18_planet.txt',
             'probs/resnet34_planet.txt', 'probs/resnet50_planet.txt']
    pred_csv(np.random.randn(2, 6), 0, 0)
