import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import fbeta_score
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
import pandas as pds
from datasets import *
import torch
import os
import glob
from planet_models.resnet_planet import resnet14_planet
from planet_models.simplenet_v2 import SimpleNetV2

BEST_THRESHOLD= [ 0.2205 , 0.0985 , 0.2495 , 0.2495,  0.281,   0.2055 , 0.0965 , 0.1695 , 0.2055,
  0.147 ,  0.1135,  0.218  , 0.131  , 0.134 ,  0.1665,  0.1035 , 0.088 ]

BEST_THRESHOLD_SINGLE = [ 0.172,  0.092,  0.221,  0.22,   0.391,  0.203,  0.161,  0.163 , 0.2  ,  0.25,
  0.165,  0.218  ,0.191  ,0.16 ,  0.19 ,  0.201 , 0.069]


def evaluate(model, image):
    """Evaluate the model given evaluation images and labels"""
    model.eval()
    if torch.cuda.is_available():
        image = image.cuda()
    image = Variable(image, volatile=True)
    output = model(image)
    return output


def split_train_validation(num_val=3000):
    """
    Save train image names and validation image names to csv files
    """
    train_image_idx = np.sort(np.random.choice(40479, 40479-3000, replace=False))
    all_idx = np.arange(40479)
    validation_image_idx = np.zeros(num_val, dtype=np.int32)
    val_idx = 0
    train_idx = 0
    for i in all_idx:
        if not i in train_image_idx:
            validation_image_idx[val_idx] = i
            val_idx += 1
        else:
            train_idx += 1
    # save train
    train = []
    for name in train_image_idx:
        train.append('train_%s' % name)

    eval = []
    for name in validation_image_idx:
        eval.append('train_%s' % name)

    df = pds.DataFrame(train)
    df.to_csv('train.csv', index=False, header=False)

    df = pds.DataFrame(eval)
    df.to_csv('validation.csv', index=False, header=False)


def threshold_labels(y, threshold=0.2):
    """
        y is a numpy array of shape N, num_classes, threshold can either be a float or a numpy array
    """

    if hasattr(threshold, '__iter__'):
        for i in range(y.shape[-1]):
            y[:, i] = (y[:, i] > threshold[i]).astype(np.int)
    else:
        y[y >= threshold] = 1
        y[y <= threshold] = 0
    return y


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

        plt.savefig('../log/%s_losses.jpg' % self.name)

        plt.figure()
        plt.plot(np.arange(len(f2_scores)), f2_scores)
        plt.savefig('../log/%s_fcscore.jpg' % self.name)

        plt.close('all')

