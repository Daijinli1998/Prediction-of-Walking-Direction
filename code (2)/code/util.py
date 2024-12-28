import os
import math
import pandas as pd
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image


def get_img(df, img_size=32, show=False, wait_ms=-1):
    df = df.drop_duplicates(inplace=False)
    df = df.dropna()

    datas = df.values

    if len(datas) == 0:
        datas = np.zeros((1, 3), dtype=np.uint8)
    else:
        datas = (datas - datas.min(0)) / (datas.max(0) - datas.min(0)) * (img_size - 1)
        datas = datas.astype('uint16')

    xoy_arr = np.zeros((img_size, img_size), dtype=np.uint8)
    xoz_arr = np.zeros((img_size, img_size), dtype=np.uint8)
    zoy_arr = np.zeros((img_size, img_size), dtype=np.uint8)
    for x, y in datas[:, :2]:
        xoy_arr[img_size - 1 - y][x] = 255
    for x, z in datas[:, (0, 2)]:
        xoz_arr[z][x] = 255
    for z, y in datas[:, 1:]:
        zoy_arr[img_size - 1 - z][img_size - 1 - y] = 255

    if show:
        cv2.imshow("xoy", xoy_arr)
        cv2.imshow("xoz", xoz_arr)
        cv2.imshow("zoy", zoy_arr)

    img = np.array(list(zip(xoy_arr, xoz_arr, zoy_arr))).swapaxes(1, 2)

    if show:
        cv2.imshow("img", img)
        cv2.waitKey(wait_ms)

    return img, [datas, xoy_arr, xoz_arr, zoy_arr]


class MyDataset(Dataset):
    def __init__(self, root_dir, img_size=32, trans=None):
        self.img_dir = os.path.join(root_dir, 'csv')
        label_csv = os.path.join(root_dir, 'dataset', 'label.csv')
        self.labels_list = pd.read_csv(label_csv).values.tolist()
        self.trans = trans
        self.to_tensor = transforms.ToTensor()
        self.img_size = img_size

    def __getitem__(self, index):
        file_name = os.path.join(self.img_dir, '{}.csv'.format(index))
        df = pd.read_csv(file_name)
        img, _ = get_img(df, self.img_size)
        if self.trans is not None:
            img = self.trans(img)
        label = self.labels_list[index]
        return img, label

    def __len__(self):
        return len(self.labels_list)


class MyDataset2(Dataset):
    def __init__(self, img_dir, index_list, label, img_size=32, trans=None):
        self.img_dir = img_dir
        self.imgs_index = index_list
        self.label = label
        self.trans = trans
        self.img_size = img_size

    def __getitem__(self, index):
        csv_name = '{}.csv'.format(self.imgs_index[index])
        df = pd.read_csv(os.path.join(self.img_dir,csv_name))
        img, _ = get_img(df, self.img_size)
        if self.trans is not None:
            img = self.trans(img)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.imgs_index)


class RgbDataset(Dataset):
    def __init__(self, img_dir, index_list, label, img_size=32, trans=None):
        self.img_dir = img_dir
        self.imgs_index = index_list
        self.label = label
        self.trans = trans
        self.img_size = img_size

    def __getitem__(self, index):
        img_name = '{}.jpg'.format(self.imgs_index[index])
        img = Image.open(os.path.join(self.img_dir, img_name))
        if self.trans is not None:
            img = self.trans(img)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.imgs_index)


class CnnModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CnnModel, self).__init__()
        self.model = nn.Sequential(
            # 32 x 32 x input_dim -> 32 x 32 x 32
            nn.Conv2d(input_dim, 32, 5, 1, 2),
            nn.ReLU(),
            # 32 x 32 x 32 -> 16 x 16 x 32
            nn.MaxPool2d(2),
            # 16 x 16 x 32 -> 16 x 16 x 64
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            # 16 x 16 x 64 -> 8 x 8 x 64
            nn.MaxPool2d(2),
            # 8 x 8 x 64 -> 8 x 8 x 128
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            # 8 x 8 x 128 -> 4 x 4 x 128
            nn.MaxPool2d(2),
            # 4 x 4 x 128 -> 128
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            # 128 -> output_dim
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def distance(a, b):
    return math.sqrt(sum([(A - B) ** 2 for (A, B) in zip(a, b)]))
