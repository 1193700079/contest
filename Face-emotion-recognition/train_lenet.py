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
from torch.autograd import Variable
from datetime import datetime
import timeit
import socket


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
labels = [label for label in sorted(os.listdir("/media/disk3/yrq/contest/Face-emotion-recognition/data/train"))]

with open("/media/disk3/yrq/contest/Face-emotion-recognition/label.txt",'w') as f:
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

n_epochs = 3
batch_size = 64
# batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True,num_workers=8)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True,num_workers=8)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True,num_workers=8)
print(len(train_data), len(train_loader))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1620, 50)
        self.fc2 = nn.Linear(50, 7)
    def forward(self, x):
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.size())
        x = x.view(x.shape[0], -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x

network = Net()
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
from torch.utils.tensorboard import SummaryWriter
save_dir = os.path.join('/media/disk3/yrq/contest/Face-emotion-recognition', 'run', 'run_' + str(1))
log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)
def train(nEpochs):

  network.train()
  acc = 0
  for epoch in range(nEpochs):
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(train_loader):

        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        loss = F.cross_entropy(probs, labels.long())
        preds = torch.max(probs, 1)[1]
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print("[train] Epoch: {}/{} Loss: {} Acc: {}".format( epoch+1, nEpochs, epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    if epoch_acc > acc :
        acc = epoch_acc
        print("best acc = {acc}")
        
        torch.save(network.state_dict(), '/media/disk3/yrq/contest/Face-emotion-recognition/model.pth')
        torch.save(optimizer.state_dict(), '/media/disk3/yrq/contest/Face-emotion-recognition/optimizer.pth')
        
# tensor([[0.1522, 0.1383, 0.1386, 0.1358, 0.1528, 0.1411, 0.1412],        [0.1413, 0.1450, 0.1520, 0.1376, 0.1441, 0.1301, 0.1499],        [0.1472, 0.1451, 0.1429, 0.1323, 0.1495, 0.1340, 0.1491],        [0.1487, 0.1405, 0.1450, 0.1378, 0.1564, 0.1311, 0.1405],        [0.1440, 0.1439, 0.1447, 0.1380, 0.1510, 0.1284, 0.1501],        [0.1452, 0.1452, 0.1439, 0.1312, 0.1506, 0.1338, 0.1501],        [0.1451, 0.1469, 0.1407, 0.1364, 0.1624, 0.1346, 0.1339],        [0.1469, 0.1487, 0.1439, 0.1398, 0.1509, 0.1354, 0.1344],        [0.1457, 0.1411, 0.1397, 0.1408, 0.1550, 0.1341, 0.1435],        [0.1575, 0.1348, 0.1430, 0.1375, 0.1489, 0.1412, 0.1370],        [0.1424, 0.1480, 0.1439, 0.1348, 0.1517, 0.1380, 0.1413],        [0.1590, 0.1379, 0.1371, 0.1435, 0.1467, 0.1332, 0.1426],        [0.1476, 0.1428, 0.1446, 0.1376, 0.1471, 0.1329, 0.1474],        [0.1406, 0.1475, 0.1518, 0.1380, 0.1442, 0.1279, 0.1501],        [0.1427, 0.1402, 0.1466, 0.1378, 0.1466, 0.1413, 0.1447],        [0.1457, 0.1438, 0.1417, 0.1334, 0.1502, 0.1380, 0.1472],        [0.1499, 0.1456, 0.1401, 0.1374, 0.1508, 0.1359, 0.1404],        [0.1472, 0.1446, 0.1433, 0.1328, 0.1479, 0.1352, 0.1490],        [0.1505, 0.1455, 0.1356, 0.1518, 0.1387, 0.1252, 0.1527],        [0.1474, 0.1435, 0.1480, 0.1374, 0.1438, 0.1298, 0.1501],        [0.1431, 0.1400, 0.1499, 0.1372, 0.1477, 0.1286, 0.1536],        [0.1441, 0.1475, 0.1472, 0.1336, 0.1465, 0.1355, 0.1457],        [0.1517, 0.1411, 0.1374, 0.1417, 0.1447, 0.1303, 0.1531],        [0.1401, 0.1486, 0.1432, 0.1381, 0.1489, 0.1366, 0.1445],        [0.1513, 0.1367, 0.1399, 0.1418, 0.1484, 0.1335, 0.1484],        [0.1564, 0.1343, 0.1332, 0.1403, 0.1501, 0.1396, 0.1462],        [0.1646, 0.1374, 0.1394, 0.1455, 0.1457, 0.1267, 0.1407],        [0.1425, 0.1399, 0.1476, 0.1322, 0.1499, 0.1349, 0.1530],        [0.1410, 0.1423, 0.1512, 0.1313, 0.1485, 0.1314, 0.1542],        [0.1542, 0.1469, 0.1375, 0.1361, 0.1417, 0.1380, 0.1457],        [0.1473, 0.1438, 0.1409, 0.1337, 0.1492, 0.1347, 0.1505],        [0.1569, 0.1358, 0.1399, 0.1335, 0.1437, 0.1420, 0.1481],        [0.1458, 0.1378, 0.1414, 0.1392, 0.1496, 0.1254, 0.1608],        [0.1464, 0.1434, 0.1437, 0.1332, 0.1489, 0.1376, 0.1469],        [0.1546, 0.1389, 0.1447, 0.1332, 0.1444, 0.1348, 0.1494],        [0.1433, 0.1431, 0.1509, 0.1396, 0.1453, 0.1353, 0.1425],        [0.1451, 0.1464, 0.1452, 0.1308, 0.1509, 0.1339, 0.1477],        [0.1412, 0.1450, 0.1451, 0.1320, 0.1591, 0.1393, 0.1383],        [0.1440, 0.1469, 0.1463, 0.1305, 0.1526, 0.1331, 0.1465],        [0.1526, 0.1493, 0.1411, 0.1435, 0.1479, 0.1318, 0.1338],        [0.1498, 0.1449, 0.1516, 0.1370, 0.1386, 0.1246, 0.1535],        [0.1506, 0.1399, 0.1401, 0.1390, 0.1444, 0.1371, 0.1490],        [0.1456, 0.1424, 0.1526, 0.1287, 0.1490, 0.1313, 0.1503],        [0.1588, 0.1423, 0.1397, 0.1423, 0.1365, 0.1306, 0.1496],        [0.1692, 0.1377, 0.1294, 0.1528, 0.1369, 0.1277, 0.1463],        [0.1450, 0.1422, 0.1456, 0.1400, 0.1501, 0.1354, 0.1417],        [0.1445, 0.1457, 0.1458, 0.1401, 0.1482, 0.1262, 0.1495],        [0.1559, 0.1425, 0.1424, 0.1407, 0.1516, 0.1249, 0.1420],        [0.1554, 0.1427, 0.1373, 0.1436, 0.1510, 0.1346, 0.1354],        [0.1532, 0.1367, 0.1456, 0.1405, 0.1468, 0.1215, 0.1557],        [0.1462, 0.1400, 0.1470, 0.1324, 0.1434, 0.1298, 0.1612],        [0.1375, 0.1471, 0.1483, 0.1379, 0.1431, 0.1265, 0.1596],        [0.1567, 0.1327, 0.1384, 0.1477, 0.1500, 0.1283, 0.1462],        [0.1392, 0.1304, 0.1556, 0.1299, 0.1633, 0.1384, 0.1432],        [0.1595, 0.1376, 0.1377, 0.1487, 0.1429, 0.1338, 0.1399],        [0.1586, 0.1373, 0.1395, 0.1335, 0.1456, 0.1404, 0.1450],        [0.1465, 0.1481, 0.1445, 0.1427, 0.1461, 0.1291, 0.1431],        [0.1487, 0.1463, 0.1460, 0.1361, 0.1476, 0.1307, 0.1446],        [0.1500, 0.1432, 0.1429, 0.1369, 0.1507, 0.1364, 0.1399],        [0.1507, 0.1393, 0.1481, 0.1370, 0.1386, 0.1414, 0.1449],        [0.1476, 0.1481, 0.1431, 0.1362, 0.1496, 0.1383, 0.1371],        [0.1367, 0.1458, 0.1547, 0.1330, 0.1483, 0.1356, 0.1459],        [0.1628, 0.1428, 0.1347, 0.1391, 0.1406, 0.1326, 0.1473],        [0.1639, 0.1380, 0.1373, 0.1427, 0.1427, 0.1385, 0.1369]],       grad_fn=<SoftmaxBackward>)
# tensor([[-1.8824, -1.9783, -1.9765, -1.9969, -1.8783, -1.9585, -1.9574],        [-1.9567, -1.9311, -1.8840, -1.9834, -1.9371, -2.0395, -1.8979],        [-1.9162, -1.9303, -1.9457, -2.0230, -1.9002, -2.0102, -1.9032],        [-1.9055, -1.9623, -1.9313, -1.9823, -1.8553, -2.0316, -1.9626],        [-1.9380, -1.9388, -1.9331, -1.9806, -1.8907, -2.0530, -1.8963],        [-1.9296, -1.9295, -1.9388, -2.0313, -1.8932, -2.0114, -1.8962],        [-1.9300, -1.9180, -1.9609, -1.9924, -1.8178, -2.0054, -2.0109],        [-1.9183, -1.9056, -1.9386, -1.9679, -1.8910, -1.9993, -2.0070],        [-1.9263, -1.9581, -1.9683, -1.9601, -1.8640, -2.0092, -1.9413],        [-1.8485, -2.0038, -1.9448, -1.9839, -1.9043, -1.9574, -1.9877],        [-1.9494, -1.9102, -1.9388, -2.0043, -1.8861, -1.9806, -1.9569],        [-1.8388, -1.9809, -1.9872, -1.9414, -1.9194, -2.0158, -1.9480],        [-1.9134, -1.9466, -1.9340, -1.9831, -1.9165, -2.0181, -1.9145],        [-1.9622, -1.9140, -1.8854, -1.9805, -1.9365, -2.0567, -1.8964],        [-1.9468, -1.9648, -1.9197, -1.9823, -1.9197, -1.9565, -1.9332],        [-1.9265, -1.9393, -1.9540, -2.0142, -1.8959, -1.9806, -1.9157],        [-1.8980, -1.9267, -1.9655, -1.9852, -1.8917, -1.9961, -1.9633],        [-1.9159, -1.9340, -1.9426, -2.0192, -1.9111, -2.0010, -1.9037],        [-1.8937, -1.9274, -1.9979, -1.8851, -1.9756, -2.0780, -1.8794],        [-1.9145, -1.9412, -1.9108, -1.9852, -1.9391, -2.0419, -1.8962],        [-1.9443, -1.9659, -1.8980, -1.9863, -1.9129, -2.0513, -1.8735],        [-1.9374, -1.9142, -1.9162, -2.0131, -1.9205, -1.9991, -1.9261],        [-1.8861, -1.9581, -1.9851, -1.9542, -1.9327, -2.0381, -1.8764],        [-1.9651, -1.9065, -1.9435, -1.9795, -1.9047, -1.9909, -1.9345],        [-1.8888, -1.9902, -1.9667, -1.9530, -1.9080, -2.0134, -1.9079],        [-1.8556, -2.0079, -2.0161, -1.9642, -1.8962, -1.9691, -1.9226],        [-1.8044, -1.9849, -1.9707, -1.9278, -1.9260, -2.0657, -1.9608],        [-1.9484, -1.9665, -1.9132, -2.0238, -1.8980, -2.0032, -1.8771],        [-1.9588, -1.9498, -1.8891, -2.0302, -1.9069, -2.0297, -1.8693],        [-1.8695, -1.9181, -1.9845, -1.9941, -1.9542, -1.9806, -1.9264],        [-1.9155, -1.9392, -1.9595, -2.0125, -1.9027, -2.0051, -1.8937],        [-1.8521, -1.9965, -1.9666, -2.0135, -1.9399, -1.9521, -1.9096],        [-1.9256, -1.9817, -1.9564, -1.9717, -1.8998, -2.0765, -1.8274],        [-1.9212, -1.9423, -1.9402, -2.0161, -1.9047, -1.9837, -1.9178],        [-1.8667, -1.9737, -1.9333, -2.0159, -1.9350, -2.0040, -1.9015],        [-1.9429, -1.9443, -1.8913, -1.9689, -1.9287, -2.0003, -1.9483],        [-1.9302, -1.9217, -1.9293, -2.0344, -1.8908, -2.0106, -1.9129],        [-1.9573, -1.9308, -1.9302, -2.0251, -1.8384, -1.9711, -1.9785],        [-1.9377, -1.9178, -1.9218, -2.0360, -1.8797, -2.0170, -1.9211],        [-1.8798, -1.9017, -1.9585, -1.9416, -1.9109, -2.0267, -2.0114],        [-1.8986, -1.9317, -1.8866, -1.9876, -1.9764, -2.0823, -1.8741],        [-1.8934, -1.9667, -1.9656, -1.9736, -1.9352, -1.9870, -1.9038],        [-1.9268, -1.9491, -1.8796, -2.0500, -1.9038, -2.0303, -1.8951],        [-1.8399, -1.9497, -1.9682, -1.9495, -1.9911, -2.0355, -1.8997],        [-1.7764, -1.9824, -2.0450, -1.8783, -1.9889, -2.0582, -1.9224],        [-1.9311, -1.9505, -1.9272, -1.9662, -1.8962, -1.9997, -1.9538],        [-1.9344, -1.9263, -1.9255, -1.9655, -1.9089, -2.0703, -1.9003],        [-1.8586, -1.9484, -1.9492, -1.9612, -1.8865, -2.0798, -1.9522],        [-1.8615, -1.9473, -1.9855, -1.9408, -1.8903, -2.0058, -1.9994],        [-1.8761, -1.9897, -1.9272, -1.9628, -1.9188, -2.1075, -1.8597],        [-1.9226, -1.9658, -1.9170, -2.0221, -1.9424, -2.0420, -1.8252],        [-1.9844, -1.9168, -1.9087, -1.9812, -1.9440, -2.0673, -1.8349],        [-1.8536, -2.0193, -1.9777, -1.9124, -1.8970, -2.0534, -1.9230],        [-1.9719, -2.0375, -1.8605, -2.0409, -1.8119, -1.9778, -1.9432],        [-1.8359, -1.9837, -1.9824, -1.9061, -1.9456, -2.0116, -1.9669],        [-1.8412, -1.9855, -1.9695, -2.0137, -1.9267, -1.9634, -1.9307],        [-1.9207, -1.9102, -1.9343, -1.9471, -1.9237, -2.0475, -1.9442],        [-1.9058, -1.9219, -1.9240, -1.9946, -1.9131, -2.0350, -1.9339],        [-1.8968, -1.9438, -1.9459, -1.9886, -1.8922, -1.9920, -1.9669],        [-1.8926, -1.9710, -1.9096, -1.9881, -1.9760, -1.9563, -1.9316],        [-1.9131, -1.9097, -1.9445, -1.9939, -1.8998, -1.9782, -1.9869],        [-1.9901, -1.9255, -1.8661, -2.0173, -1.9085, -1.9981, -1.9249],        [-1.8152, -1.9466, -2.0046, -1.9722, -1.9617, -2.0203, -1.9151],        [-1.8083, -1.9808, -1.9855, -1.9473, -1.9470, -1.9766, -1.9886]],       grad_fn=<LogSoftmaxBackward>) 
#tensor([4, 3, 7, 7, 6, 1, 7, 4, 1, 1, 1, 6, 6, 7, 8, 4, 1, 1, 3, 3, 7, 6, 4, 6,        4, 4, 7, 4, 1, 1, 7, 4, 3, 4, 6, 4, 6, 6, 3, 4, 7, 7, 4, 8, 6, 6, 3, 1,        8, 3, 7, 3, 8, 6, 8, 7, 6, 1, 4, 3, 4, 4, 4, 3])
          
train(100)

# def test():
#   network.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_loader:
#       output = network(data)
#       test_loss += F.nll_loss(output, target, size_average=False).item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= len(test_loader.dataset)
#   test_losses.append(test_loss)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))

  
# test()

