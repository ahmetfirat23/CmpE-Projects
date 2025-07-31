# Reference for U-Net with residuals: https://www.linkedin.com/pulse/unet-resblock-semantic-segmentation-nishank-singla/

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self, is_grayscale=False):
        channel_count = 1 if is_grayscale else 3
        super(UNET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_count, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_res_1 = nn.Conv2d(in_channels=channel_count, out_channels=64, kernel_size=1, stride=1 , padding='same')
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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 256, 256)
        self.fc2 = nn.Linear(256, 4)
        self.fc3 = nn.Linear(8,8)
        self.fc4 = nn.Linear(8, 256) 
        self.fc5 = nn.Linear(256, 8 * 8 * 256)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 , padding='same')
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1 , padding='same')
        self.bn8 = nn.BatchNorm2d(128)
        self.conv_res_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1 , padding='same')
        self.bn_res_4 = nn.BatchNorm2d(128)

        self.trans_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1 , padding='same')
        self.bn10 = nn.BatchNorm2d(64)
        self.conv_res_5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1 , padding='same')
        self.bn_res_5 = nn.BatchNorm2d(64)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=channel_count, kernel_size=3, stride=1 , padding='same')

    def forward(self, x, action):
        x_res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(x + self.bn_res_1(self.conv_res_1(x_res)))
        x_skip_1 = x
        x = self.maxpool1(x)

        x_res = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(x + self.bn_res_2(self.conv_res_2(x_res)))
        x_skip_2 = x
        x = self.maxpool2(x)

        x_res = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(x + self.bn_res_3(self.conv_res_3(x_res)))

        x = self.maxpool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, action), 1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.unflatten(x)
        x = self.upsample(x)

        x = self.trans_conv1(x)
        x = torch.cat((x, x_skip_2), dim=1)
        x_res = x
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(x + self.bn_res_4(self.conv_res_4(x_res)))

        x = self.trans_conv2(x)
        x = torch.cat((x, x_skip_1), dim=1)
        x_res = x
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(x + self.bn_res_5(self.conv_res_5(x_res)))

        x = self.conv11(x)
        return x
