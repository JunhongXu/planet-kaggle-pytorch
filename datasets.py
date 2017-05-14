import os
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from labels import *
from skimage import io


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'jpg', '.jpeg'])


def load_img(filepath):
    """
        This function reads two types of image:
            1. If it is a .jpg, it uses PIL to open and read.
            2. If it is a .tif, it uses tifffile to open it.
    """
    if is_image_file(filepath):
        image = Image.open(filepath)
        image = image.convert('RGB')
    elif '.tif' in filepath:
        image = io.imread(filepath)
    else:
        raise OSError('File is not either a .tif file or an image file.')
    return image


def input_transform(crop_size):
    return Compose(
        [
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )


class PlanetDataSet(Dataset):
    def __init__(self, image_dir, label_dir=None, num_labels=17, mode='Train', input_transform=ToTensor(),
                 target_transform=None, tif=False):
        super(PlanetDataSet, self).__init__()
        self.mode = mode
        self.tif = tif
        self.images = []
        suffix = '.jpg' if tif is False else '.tif'
        print('[*]Loading Dataset {}'.format(image_dir))
        t = time.time()
        if mode == 'Train' or mode == 'Validation':
            self.targets = []
            self.labels = pd.read_csv(label_dir)
            image_names = pd.read_csv('../dataset/train.csv' if mode == 'Train' else '../dataset/validation.csv')
            image_names = image_names.as_matrix().flatten()
            self.image_filenames = image_names
            for image in image_names:
                str_target = self.labels.loc[self.labels['image_name'] == image]
                image = os.path.join(image_dir, '{}{}'.format(image, suffix))
                target = np.zeros(num_labels, dtype=np.float32)
                target_index = [label_to_idx[l] for l in str_target['tags'].values[0].split(' ')]
                target[target_index] = 1
                image = load_img(image)
                self.images.append(image)
                self.targets.append(target)
        elif mode == 'Test':
            self.image_filenames = sorted([os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
                                           if is_image_file(filename)])
            for image in self.image_filenames:
                image = load_img(image)
                self.images.append(image)

        print('[*]Dataset loading completed, total time elisped {}'.format(time.time() - t))
        print('[*]Total number of data is {}'.format(len(self)))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'Test':
            image = load_img(self.image_filenames[index])
            if '.tif' in self.image_filenames[index]:
                im_id = self.image_filenames[index].split('/')[-1].strip('.tif')
            else:
                im_id = self.image_filenames[index].split('/')[-1].strip('.jpg')
            if self.input_transform is not None:
                image = self.input_transform(image)
            return image, im_id
        else:
            image = self.images[index]
            target = self.targets[index]
            if self.input_transform is not None:
                image = self.input_transform(image)
            return image, torch.from_numpy(target)

    def __len__(self):
        return len(self.image_filenames)


def train_tif_loader(batch_size=64, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train-tif',
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train.csv',
        mode='Train',
        input_transform=transform,
        tif=True
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, )


def validation_tif_loader(batch_size=64, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train-tif',
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train.csv',
        mode='Validation',
        input_transform=transform,
        tif=True
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, )


def test_tif_loader(batch_size=64, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/test/test-tif',
        mode='Test',
        input_transform=transform,
        tif=True
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, )


def train_jpg_loader(batch_size=64, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train-jpg',
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train.csv',
        mode='Train',
        input_transform=transform
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, )


def validation_jpg_loader(batch_size=64, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train-jpg',
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/train/train.csv',
        mode='Validation',
        input_transform=transform
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def test_jpg_loader(batch_size=128, transform=ToTensor()):
    dataset = PlanetDataSet(
        '/media/jxu7/BACK-UP/Data/AmazonPlanet/test/test-jpg',
        mode='Test',
        input_transform=transform
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


