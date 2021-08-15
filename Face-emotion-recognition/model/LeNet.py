import torch.nn.functional as F
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self,num_classes = None):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1620, 50)
        self.fc2 = nn.Linear(50, num_classes)
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