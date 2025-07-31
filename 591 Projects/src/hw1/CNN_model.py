import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1 , padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16 * 16 * 64, 256) # symbolic representation
        self.fc2 = nn.Linear(256, 4)
        self.fc3 = nn.Linear(8, 8) # actions
        self.fc4 = nn.Linear(8, 256)
        self.fc5 = nn.Linear(256, 2)


    def forward(self, x, action):
        # x = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.downsample(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.downsample(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.downsample(x)
        # x = self.pool(x)
        x = self.flatten(x)
#        x = x.view(-1, 16 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, action), 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
