from __future__ import print_function, division
import pandas as pd
from skimage import transform
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
from tqdm import tqdm
import os
import re
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import adjust_gamma, adjust_contrast, adjust_brightness
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class train_dataset(Dataset):
    def __init__(self, config, transform=None):
        self.CONFIG = config
        self.ROOT_PATH = self.CONFIG['DIR']['ROOT_PATH']
        self.IMG_PATH = os.path.join(self.ROOT_PATH, self.CONFIG['DIR']['IMG_FOLDER'])
        self.LABEL_PATH = os.path.join(self.ROOT_PATH, self.CONFIG['DIR']['LABEL_FOLDER'])
        self.TXT_PATH = os.path.join(self.ROOT_PATH, config['DIR']['TRAIN_TXT'])
        self.IMG_LIST = open(self.TXT_PATH, 'r')
        self.IMG_LIST = self.IMG_LIST.readlines()
        self.IMG_LIST = list(map(self.remove_backslash, self.IMG_LIST))
        self.LABEL_LIST = list(map(self.get_label_path, self.IMG_LIST))
        self.transform = transform

    def remove_backslash(self,string):
        return re.sub('\n','',string)

    def get_label_path(self, string):
        string = re.sub('.jpg', '.txt',string)
        return re.sub(self.CONFIG['DIR']['IMG_FOLDER'], self.CONFIG['DIR']['LABEL_FOLDER'], string)

    def __len__(self):
        return len(self.IMG_LIST)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img = cv2.imread(self.IMG_LIST[idx])
        img = Image.open(self.IMG_LIST[idx])
        label = open(self.LABEL_LIST[idx], 'r')
        label = label.readline()
        label = int(label.split(' ')[0])-1
        sample = {'image':img, 'label':label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # sample['label'] = torch.Tensor(sample['label'])

        return sample


class valid_dataset(train_dataset):
    def __init__(self, config, transform):
        super(valid_dataset, self).__init__(config)
        self.TXT_PATH = os.path.join(self.ROOT_PATH, self.CONFIG['DIR']['VALID_TXT'])

class TradeMarkDataset(Dataset):
    def __init__(self, batch_size, num_workers, config):
        self.config = config
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            # torchvision.transforms.RandomCrop(size=(224, 224), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize(416)
            # Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = train_dataset(self.config, transform=train_transform)
        valid_set = valid_dataset(self.config, transform=test_transform)

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def _get_statistics(self):
        train_set = train_dataset(self.config, transform=transforms.ToTensor())
        data = torch.cat([d['image'] for d in DataLoader(train_set)])

        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class devideDataset:
    def __init__(self, csv_path, train_ratio):
        """
        Args:
            :param csv_path: label정보가 있는 csv 파일의 경로(string)
            :param train_ratio: 전체 데이터 셋 중에서 학습 데이터가 차지할 비중(float(ex:0.7))
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, engine='python')
        self.df = self.df.iloc[:,0:2]
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.train_ratio = train_ratio
        self.makePaths()
        if not os.path.exists(self.train_csv_path):
            self.devide()
            self.save()

    def makePaths(self):
        total_csv_path = self.csv_path.split('/')
        root_dir = '/'.join(total_csv_path[:len(total_csv_path)-1])
        filename = total_csv_path[len(total_csv_path)-1]
        train_filename = f'train_{filename}'
        test_filename = f'test_{filename}'
        self.train_csv_path = os.path.join(root_dir, train_filename)
        self.test_csv_path = os.path.join(root_dir, test_filename)

    def devide(self):
        data_length = len(self.df.iloc[:,0])
        train_length = int(data_length * self.train_ratio)
        self.train_df = self.df.iloc[:train_length,:]
        self.test_df = self.df.iloc[train_length:,:]

    def save(self):
        self.train_df.to_csv(self.train_csv_path, index=False)
        self.test_df.to_csv(self.test_csv_path, index=False)


class returnMLB():
    def __init__(self, csv_path, discriminator):
        self.df = pd.read_csv(csv_path)
        self.discriminator = discriminator

    def returnMLB(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = np.unique(np.array(label_list))
        multilabel_list = [label.split(self.discriminator) for label in label_list]
        multilabel_list = [tuple(label) for label in multilabel_list]
        mlb = MultiLabelBinarizer()
        mlb.fit(multilabel_list)
        return mlb

class readDataset(Dataset):
    def __init__(self, csv_path, img_dir, mlb, discriminator, transform=None):
        """
        Args:
            data (string): label정보가 있는 csv 파일의 경로(string)
            img_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            disciriminator: label의 구분자(만약 하나의 라벨이 'black_jeans'라면 discriminator는 '_'를 의미함)
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, engine='python')
        self.root_dir = img_dir
        self.transform = transform
        self.w_p_path = re.sub('.csv', '_w_p.npy', self.csv_path)
        self.w_n_path = re.sub('.csv', '_w_n.npy', self.csv_path)
        self.onehot_path = re.sub('.csv', '_onehot.npy', self.csv_path)
        self.discriminator = discriminator
        self.MLB = mlb
        if os.path.isfile(self.onehot_path):
            self.df['onehot'] = np.load(self.onehot_path).tolist()
            self.df['weight_p'] = np.load(self.w_p_path).tolist()
            self.df['weight_n'] = np.load(self.w_n_path).tolist()
        else:
            self.updateDf()

    def updateDf(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = [tuple(label.split(self.discriminator)) for label in label_list]
        self.df.iloc[:,1] = label_list
        onehot_list = []
        P = self.returnP()
        W_P = []
        W_N = []
        print('Calculating positive weights and negative weights')
        for label in tqdm(label_list):
            onehot = np.ndarray.flatten(self.MLB.transform([label]))
            onehot_list.append(onehot)
            w_p = []
            w_n = []
            for i, p in zip(onehot,P):
                if i == 1:
                    w_p.append(np.exp(1-p))
                    w_n.append(0.)
                else:
                    w_p.append(0.)
                    w_n.append(np.exp(p))
            W_P.append(np.array(w_p))
            W_N.append(np.array(w_n))
        self.df['onehot'] = onehot_list
        self.df['weight_p'] = W_P
        self.df['weight_n'] = W_N
        self.df.columns = ['img_name', 'tags', 'onehot', 'weight_p', 'weight_n']

    def returnP(self):
        label_list = self.df.iloc[:, 1].tolist()
        P = np.zeros(len(self.MLB.classes_))
        print('Calculationg P')
        for i, label in enumerate(tqdm(label_list)):
            label = [self.df.iloc[i,1]]
            result = np.ndarray.flatten(self.MLB.transform(label))
            P = P+result
        return P / np.sum(P)

    def returnOneHotMultiLabel(self, idx):
        label = [self.df.iloc[idx,1]]
        result = np.ndarray.flatten(self.MLB.transform(label))
        result = np.array(result, dtype='float').flatten()
        return result

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        # image = np.array(image, dtype='float') / 255.0 #0.~1. 사이의 값으로 Normalizing

        label = self.df.iloc[idx, 2]
        weight_p = self.df.iloc[idx,3]
        weight_n = self.df.iloc[idx,4]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}

        return sample

class readTestDataset(Dataset):
    def __init__(self, csv_path, img_dir, mlb, discriminator, transform=None):
        """
        Args:
            data (string): label정보가 있는 csv 파일의 경로(string)
            img_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            disciriminator: label의 구분자(만약 하나의 라벨이 'black_jeans'라면 discriminator는 '_'를 의미함)
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, engine='python')
        self.root_dir = img_dir
        self.transform = transform
        self.discriminator = discriminator
        self.MLB = mlb
        self.updateDf()

    def updateDf(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = [tuple(label.split(self.discriminator)) for label in label_list]
        self.df.iloc[:,1] = label_list
        onehot_list = []
        for label in tqdm(label_list):
            onehot = np.ndarray.flatten(self.MLB.transform([label]))
            onehot_list.append(onehot)

        self.df['onehot'] = onehot_list
        self.df.columns = ['FILENAME', 'LABEL', 'onehot']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = np.array(image, dtype='float') / 255.0 #0.~1. 사이의 값으로 Normalizing
        label = self.df.iloc[idx, 2]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class AdjustGamma(object):
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, 0.8, gain=1)

class AdjustContrast(object):
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, 2)

class AdjustBrightness(object):
    def __call__(self, img):
        return transforms.functional.adjust_brightness(img, 2)

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    AdjustGamma(),
    AdjustContrast(),
    AdjustBrightness(),
    PyTMinMaxScalerVectorized()
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    PyTMinMaxScalerVectorized()
])

class returnDataLoader:
    def __init__(self, csv_path, train_ratio, img_dir, discriminator):
        self.DATASET = devideDataset(csv_path, train_ratio)
        self.TRAIN_CSV_PATH = self.DATASET.train_csv_path
        self.TEST_CSV_PATH = self.DATASET.test_csv_path
        self.MLB = returnMLB(csv_path, discriminator).returnMLB()
        self.TRAIN_DATASET = readDataset(self.TRAIN_CSV_PATH,
                                         img_dir,
                                         self.MLB,
                                         discriminator,
                                         train_transforms)
        self.VAL_DATASET = readDataset(self.TEST_CSV_PATH,
                                       img_dir,
                                       self.MLB,
                                       discriminator,
                                       valid_transforms)
        self.TRAIN_DATA_LENGTH = len(self.TRAIN_DATASET.df.iloc[:,0])
        self.VAL_DATA_LENGTH = len(self.VAL_DATASET.df.iloc[:,0])

class testToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': image, 'label': label}