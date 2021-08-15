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
# from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
# from model import Net
# from vit_pytorch.efficient import ViT
from torch.autograd import Variable
from datetime import datetime
import timeit
import socket
# from model.LeNet5 import LeNet5
# from model.resnet18 import resnet18
from model.Res18Feature import Res18Feature
import cv2
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model  = Res18Feature()
criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
# (权重衰减 w参数的权重 )weight_decay=0.001 就是L2正则化!  weight_decay默认是0
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9)
# optimizer = optim.Adam(train_params, lr=lr, , weight_decay=5e-4)
#  每10个epoch lr变为原来的1/10 gamma 默认就是0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                        gamma=0.1)  # the scheduler divides the lr by 10 ever
model.load_state_dict(torch.load("/media/disk3/yrq/contest/Face-emotion-recognition/model3.pth",map_location=lambda storage, loc: storage))
optimizer.load_state_dict(torch.load("/media/disk3/yrq/contest/Face-emotion-recognition/optimizer3.pth",map_location=lambda storage, loc: storage))

model.to(device)
criterion.to(device)
print(model)

labels = [label for label in sorted(os.listdir("/media/disk3/yrq/contest/Face-emotion-recognition/data/train"))]
dic= dict(zip(range(len(labels)),labels))

res = []
res.append("name,label")
test_dir_path = "/media/disk3/yrq/contest/Face-emotion-recognition/data/test"

# transform1 = transforms.Compose([
# 	# transforms.CenterCrop((224,224)), # 只能对PIL图片进行裁剪
# 	transforms.ToTensor(), 
# 	]
# )
transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
model.eval()
for img in sorted(os.listdir(test_dir_path)):
    image = cv2.imread(os.path.join(test_dir_path,img))
    image = image[:, :, ::-1]
    image = transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    # img_PIL = Image.open(os.path.join(test_dir_path,img))
    # img_transformed = transform1(img_PIL)
    # img_transformed = img_transformed.unsqueeze(0)
    # img_transformed =img_transformed.to(device)
    with torch.no_grad():
        _,pred = model(image)
        probs = F.softmax(pred)
        pred = torch.argmax(probs,dim=1).item()
        # print(pred)
        ans = f"{img},{dic[pred]}"
        print(ans)
        res.append(ans)

with open("submit3.csv","w") as f:
    for i in res:
        f.write(i +"\n")