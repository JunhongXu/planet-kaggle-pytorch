from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import random
import math
import torch

KAGGLE_DATA_DIR ='/media/jxu7/BACK-UP/Data/AmazonPlanet'
CLASS_NAMES=[
    'clear',    	 # 0
    'haze',	         # 1
    'partly_cloudy', # 2
    'cloudy',	     # 3
    'primary',	     # 4
    'agriculture',   # 5
    'water',	     # 6
    'cultivation',	 # 7
    'habitation',	 # 8
    'road',	         # 9
    'slash_burn',	 # 10
    'conventional_mine', # 11
    'bare_ground',	     # 12
    'artisinal_mine',	 # 13
    'blooming',	         # 14
    'selective_logging', # 15
    'blow_down',	     # 16
]


class KgForestDataset(Dataset):

    def __init__(self, split, transform=None, height=256, width=256, label_csv='train.csv'):
        class_names = CLASS_NAMES
        num_classes = len(class_names)
        data_dir = KAGGLE_DATA_DIR
        ext = 'train' if label_csv else 'test'
        # read names
        list = data_dir +'/split/'+ split
        print(data_dir)
        with open(list) as f:
            names = f.readlines()
        names = [x.strip()for x in names]
        num = len(names)

        #read images
        images = np.zeros((num,height,width,3),dtype=np.float32)
        for n in range(num):
            img_file = data_dir + '/{}/'.format(ext) + names[n]
            jpg_file = img_file.replace('<ext>','jpg')
            image_jpg = cv2.imread(jpg_file,1)
            h,w = image_jpg.shape[0:2]
            if height!=h or width!=w:
                image_jpg = cv2.resize(image_jpg,(height,width))

            images[n,:,:,:3]=image_jpg.astype(np.float32)/255.0

        #read labels
        df = None
        labels = None
        if label_csv is not None:
            labels = np.zeros((num,num_classes),dtype=np.float32)

            csv_file = data_dir + '/train/' + label_csv   # read all annotations
            df = pd.read_csv(csv_file)
            for c in class_names:
                df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

            df1 = df.set_index('image_name')
            for n in range(num):
                shortname = names[n].split('/')[-1].replace('.<ext>','')
                labels[n] = df1.loc[shortname].values[1:]
        #save
        self.transform = transform
        self.num = num
        self.split = split
        self.names = names
        self.images = images

        self.class_names = class_names
        self.df = df
        self.labels = labels

    def __getitem__(self, index):

        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, index, 0

        else:
            label = self.labels[index]
            return img, label, index

    def __len__(self):
        return len(self.images)


################################# Transformations begin here ################################
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img, 0)  #np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img


def randomHorizontalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img


def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1,0,2)  #cv2.transpose(img)
    return img


#http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees
def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle=random.randint(1,3)*90
        if angle == 90:
            img = img.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(img,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            img = img.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(img,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]
    return img


def randomRotate(img, u=0.25, limit=90):
    if random.random() < u:
        angle = random.uniform(-limit,limit)  #degree

        height,width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width/2,height/2),angle,1.0)
        img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        #img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    return img



def randomShift(img, u=0.25, limit=4):
    if random.random() < u:
        dx = round(random.uniform(-limit,limit))  #pixel
        dy = round(random.uniform(-limit,limit))  #pixel

        height,width,channel = img.shape
        img1 =cv2.copyMakeBorder(img, limit+1, limit+1, limit+1, limit+1,borderType=cv2.BORDER_REFLECT_101)
        y1 = limit+1+dy
        y2 = y1 + height
        x1 = limit+1+dx
        x2 = x1 + width
        img = img1[y1:y2,x1:x2,:]

    return img


def randomShiftScale(img, u=0.25, limit=4):
    if random.random() < u:
        height,width,channel = img.shape
        assert(width==height)
        size0 = width
        size1 = width+2*limit
        img1  = cv2.copyMakeBorder(img, limit, limit, limit, limit,borderType=cv2.BORDER_REFLECT_101)
        size  = round(random.uniform(size0,size1))


        dx = round(random.uniform(0,size1-size))  #pixel
        dy = round(random.uniform(0,size1-size))


        y1 = dy
        y2 = y1 + size
        x1 = dx
        x2 = x1 + size

        if size ==size0:
            img = img1[y1:y2,x1:x2,:]
        else:
            img = cv2.resize(img1[y1:y2,x1:x2,:],(size0,size0),interpolation=cv2.INTER_LINEAR)

    return img


def randomShiftScaleRotate(img, u=0.5, shift_limit=4, scale_limit=0.1, rotate_limit=45):
    if random.random() < u:
        height,width,channel = img.shape

        angle = random.uniform(-rotate_limit,rotate_limit)  #degree
        scale = random.uniform(1-scale_limit,1+scale_limit)
        dx    = round(random.uniform(-shift_limit,shift_limit))  #pixel
        dy    = round(random.uniform(-shift_limit,shift_limit))

        cc = math.cos(angle/180*math.pi)*(scale)
        ss = math.sin(angle/180*math.pi)*(scale)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img


def cropCenter(img, height, width):

    h,w,c = img.shape
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2,x1:x2,:]

    return img


def toTensor(img):
    img = img.transpose((2,0,1)).astype(np.float32)
    tensor = torch.from_numpy(img).float()
    return tensor
