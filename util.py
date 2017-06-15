import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import fbeta_score
from torch.nn import functional as F
from matplotlib import pyplot as plt
import pandas as pds
from datasets import *
import torch
import os
from data.kgdataset import KgForestDataset
from planet_models.densenet_planet import densenet169, densenet121, densenet161
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet


def save_results(models, dataloader):
    """Given model/models, this function saves the result of F.sigmoid(model(x))"""
    for model in models:
        name = str(model).split()[1]
        # create
        # net = model()
        # net = nn.DataParallel(net.cuda())# nn.DataParallel(densenet169())
        # net.load_state_dict(torch.load('models/%s.pth' % name)['state_dic'
        net = torch.load('models/%s.pth' % name)
        net.eval()
        # model = nn.DataParallel(model.cuda())

        # load

        # forward
        result = []
        for i, (image, target, index) in enumerate(dataloader):
            image = Variable(image.cuda(), volatile=True)
            # N * 17
            probs = F.sigmoid(net(image))
            result.append(probs.data.cpu().numpy())

        # concatenate the probabilities
        result = np.concatenate(result)
        # save the probabilities into model.txt file
        np.savetxt(fname='probs/{}.txt'.format(name), X=result)


def optimize_threshold(fnames, labels, resolution):
    """This function optimizes threshold given dataset and probability files."""

    results = []
    for f in fnames:
        # open the file
        with open(f) as file:
            lines = file.read().split('\n')[:-1]
            N = len(lines)
            result = np.empty((N, 17))
            for index, line in enumerate(lines):
                result[index] = np.fromstring(line, dtype=np.float32, sep=' ')

        results.append(result)

    results = np.asarray(results)
    results = results.mean(axis=0)
    print(results.shape)

    # optimize threshold, labels N * 17
    threshold = [0.15] * 17
    for i in range(17):
        best_thresh = 0.0
        best_score = 0.0
        for r in range(resolution):
            r /= resolution
            threshold[i] = r
            # labels = get_labels(pred, threshold)
            preds = (results > threshold).dtype(np.int32)
            score = f2_score(preds, labels)
            if score > best_score:
                best_thresh = r
                best_score = score
        threshold[i] = best_thresh
        print(i, best_score, best_thresh)
    print('{}: {}'.format(best_score, best_thresh))
    return best_thresh


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
    # a = np.random.randn(100, 17)
    # np.savetxt('probs/model_1.txt', a)
    # optimize_threshold(['probs/model_1.txt'], 'data')
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
    dataloader = DataLoader(validation)
    save_results([resnet18_planet, resnet34_planet, resnet50_planet,
        densenet121, densenet169, densenet161,], dataloader)
