import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch

class ResidualCNN(nn.Module):
    def __init__(self, NUM_OF_FRAMES, N_ACTIONS):
        super(ResidualCNN, self).__init__()
        self.NUM_OF_FRAMES = NUM_OF_FRAMES
        self.N_ACTIONS = N_ACTIONS
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=NUM_OF_FRAMES, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_res_1 = nn.Conv2d(in_channels=NUM_OF_FRAMES, out_channels=64, kernel_size=1, stride=1 , padding='same')
        self.bn_res_1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1 , padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1 , padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_res_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1 , padding='same')
        self.bn_res_2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1 , padding='same')
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1 , padding='same')
        self.bn6 = nn.BatchNorm2d(256)
        self.conv_res_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1 , padding='same')
        self.bn_res_3 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(8 * 8 * 256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, N_ACTIONS)
        print(summary(self, input_size=(NUM_OF_FRAMES, 128, 128)))

    def forward(self, x):
        x = x.view(-1, self.NUM_OF_FRAMES, 128, 128)
        batch_size = x.size(0)
        x = self.maxpool1(self.bn2(self.conv2(self.bn1(self.conv1(x))) + self.bn_res_1(self.conv_res_1(x))))
        x = self.maxpool2(self.bn4(self.conv4(self.bn3(self.conv3(x))) + self.bn_res_2(self.conv_res_2(x))))
        x = self.maxpool3(self.bn6(self.conv6(self.bn5(self.conv5(x))) + self.bn_res_3(self.conv_res_3(x))))
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ExampleCNN(nn.Module):
    def __init__(self, NUM_OF_FRAMES, N_ACTIONS):
        super(ExampleCNN, self).__init__()
        self.NUM_OF_FRAMES = NUM_OF_FRAMES
        self.N_ACTIONS = N_ACTIONS
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(NUM_OF_FRAMES, 32, 4, 2, 1)  # (-1, 3, 128, 128) -> (-1, 32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # (-1, 32, 64, 64) -> (-1, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # (-1, 64, 32, 32) -> (-1, 128, 16, 16)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # (-1, 128, 16, 16) -> (-1, 256, 8, 8)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)  # (-1, 256, 8, 8) -> (-1, 512, 4, 4)
        self.avg = nn.AdaptiveAvgPool2d((2, 3))  # average pooling over the spatial dimensions  (-1, 512, 4, 4) -> (-1, 512),
        self.fc1 = nn.Linear(512 * 2 * 3, 512)
        self.fc2 = nn.Linear(512 , 64)
        self.fc3 = nn.Linear(64, N_ACTIONS)
        print(summary(self, input_size=(NUM_OF_FRAMES, 128, 128)))

    def forward(self, x):
        x = x.view(-1, self.NUM_OF_FRAMES, 128, 128)
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.avg(x)
        x = x.view(batch_size, -1)        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self, NUM_OF_FRAMES, N_ACTIONS):
        self.NUM_OF_FRAMES = NUM_OF_FRAMES
        self.N_ACTIONS = N_ACTIONS
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(NUM_OF_FRAMES * 6 , 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, N_ACTIONS)
        print(summary(self, input_size=(NUM_OF_FRAMES, 6)))

    def forward(self, x):
        x = x.view(-1, self.NUM_OF_FRAMES * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x