import pandas as pd
import numpy as np
import glob
from data import kgdataset
from data.kgdataset import KgForestDataset

kgdataset.KAGGLE_DATA_DIR = '../../../kaggle'
from util import name_idx, pred_csv

filenames = glob.glob('submissions/*.csv')
print('\n'.join(filenames))

# read all csv files and convert tags to labels
dfs =[pd.read_csv(name) for name in filenames]
labels = np.empty((8, 61191, 17))
for df_idx, df in enumerate(dfs):
    print(df_idx)
    for row, tag in enumerate(df['tags']):
        label = np.zeros(17)
        idx = [name_idx()[name] for name in tag.split(' ')]
        label[idx] = 1
        labels[df_idx, row, :] = label

majority_voting = labels.sum(axis=0)
majority_voting = (majority_voting >= 4).astype(int)

pred_csv(majority_voting, name='sub_ensembles')