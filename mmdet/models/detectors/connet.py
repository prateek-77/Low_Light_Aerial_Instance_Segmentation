import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ConNet(nn.Module):

    def __init__(self):
        super(ConNet, self).__init__()

        self.conv0 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv = nn.Conv2d(16, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 3, padding=1)
        self.init_weight()

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x1 = x
        x = F.relu(self.conv(x))
        x2 = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x + x2
        x = F.relu(self.conv3(x))
        x = x + x1
        x = F.relu(self.conv4(x))
        
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

def connet():
    return ConNet()
