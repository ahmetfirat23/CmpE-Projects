import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 1, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4) # symbolic representation
        self.fc4 = nn.Linear(8, 32) # actions
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x, action):
        x = self.downsample(x)
        x = x.view(-1, 64 * 64 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.cat((x, action), 1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
