import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ..losses.illuminate import lll
from mmdet.core import auto_fp16, force_fp32

class DecomNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        print(channel)
        self.conv0 = nn.Conv2d(4, channel//2, kernel_size, padding=1)
        self.conv = nn.Conv2d(4, channel, kernel_size*3, padding=4)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channel, channel*2, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channel*2, channel*2, kernel_size, padding=1)
        self.conv4 = nn.ConvTranspose2d(channel*2, channel, kernel_size, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channel*2, channel, kernel_size, padding=1)
        self.conv6 = nn.Conv2d(3*int(channel/2), channel, kernel_size, padding=1)
        self.conv7 = nn.Conv2d(channel, 4, kernel_size, padding=1)
        self.upsample = F.upsample_nearest
        self.criterion = lll()

    def forward(self, x):

        x_hist = torch.max(x, dim=1, keepdim=True)
        #x_hist = x_hist.float()
        x = torch.cat((x, x_hist[0]), dim=1)

        x1 = F.relu(self.conv0(x))
        x = self.conv(x)
        x = F.relu(self.conv1(x))
        y = x
        shp = y.data.shape
#        print("works!")
        shp = shp[2:4]
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.upsample(x, size=shp)
        x = F.relu(self.conv4_1(x))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.conv5(x))
        
        x = torch.cat((x, x1), dim=1)
        x = self.conv6(x)
        out = self.conv7(x)

        return out

    @force_fp32(apply_to=('y', ))
    def loss(self, input_im, y):
        R = F.sigmoid(y[:,0:3,:,:])
        L = F.sigmoid(y[:,3:4,:,:])
        loss_illumination = self.criterion(input_im, R, L)
        loss_illumination = 0.8 * loss_illumination
        return loss_illumination

def build_illenh(channel=64, kernel_size=3):
    return DecomNet(channel, kernel_size)
