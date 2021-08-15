from __future__ import print_function

import glob
from itertools import chain
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms



train_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


class FaceDataset(Dataset):
    def __init__(self, file_list = None, transform=None):
        self.file_list = file_list
        self.transform = transform
        labels = [label for label in sorted(os.listdir("/media/disk3/yrq/contest/Face-emotion-recognition/data/train"))]
        self.dic= dict(zip(range(len(labels)),labels))
        # self.dic= dict(zip(labels,range(len(labels))))

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):

        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        # label = img_path.split('/')[-2]
        # label_idx = np.array(self.dic[label] )
        labels = np.array(self.dic[idx])
        return img_transformed, torch.from_numpy(label_idx)