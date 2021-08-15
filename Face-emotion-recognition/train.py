from __future__ import print_function
import sys
sys.path.append("/media/disk3/yrq/contest/Face-emotion-recognition")
import glob
from itertools import chain
import os
import random
import zipfile

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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
# from model import Net
from vit_pytorch.efficient import ViT


# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# device = 'cuda:0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dir = '/media/disk3/yrq/contest/Face-emotion-recognition/data/train'
test_dir = '/media/disk3/yrq/contest/Face-emotion-recognition/data/test'


train_list = os.listdir(train_dir)
train_list.sort()
labels = [label for label in train_list]

with open("label.txt",'w') as f:
    for i , label in enumerate(labels):
        f.write(f"{label} {i}\n")

train_list = glob.glob(os.path.join(train_dir, '*' ,'*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))



print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                        #   stratify=labels,
                                          random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

train_transforms = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
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

# '/media/disk3/yrq/contest/Face-emotion-recognition/data/test/04798.png'
# '/media/disk3/yrq/contest/Face-emotion-recognition/data/train/disgusted/im178.png'
# for label in sorted(os.listdir(train_dir)):
#     for fname in os.listdir(os.path.join(train_dir, label)):       
#         labels.append(label)
# label2index = {label: index+1 for index, label in enumerate(sorted(set(labels)))}
# label_array = np.array([label2index[label] for label in labels], dtype=int)

dic= dict(zip(labels,range(len(labels))))
class FaceDataset(Dataset):
    def __init__(self, file_list = None, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):

        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split('/')[-2]
        label_idx = np.array(dic[label] )
        # labels = np.array(label_array[idx])
        return img_transformed, torch.from_numpy(label_idx)


train_data = FaceDataset(train_list, transform=train_transforms)
valid_data = FaceDataset(valid_list, transform=test_transforms)
test_data = FaceDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))


efficient_transformer = Linformer(
    dim=64,
    seq_len=16+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=64,
    image_size=48,
    patch_size=12,
    num_classes=7,
    transformer=efficient_transformer,
    channels=1,
).to(device)
print(model)

class Net(nn.Module):
    def __init__(self , num_classes =None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net(num_classes = 7)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

