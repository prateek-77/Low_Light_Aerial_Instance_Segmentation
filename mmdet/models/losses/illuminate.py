import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pylab import *
from PIL import ImageFilter
from ..registry import LOSSES

def histeq(im,nbr_bins = 256):
    imhist,bins = histogram(im.flatten(),nbr_bins,density=True)
    cdf = imhist.cumsum()
    #print("works")
    cdf = 1.0*cdf / cdf[-1]
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

def rgb_to_grayscale(tensor):
    tensor = tensor.cpu()
    img = transforms.functional.to_pil_image(tensor, mode=None)
    img_gs = transforms.functional.to_grayscale(img, num_output_channels=1)
    return transforms.functional.to_tensor(img_gs)

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32, device='cuda:0'), [2, 1, 2, 1])
    smooth_kernel_y = smooth_kernel_x.permute(2,1,0,3)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    return torch.abs(F.conv2d(input_tensor, kernel, stride=(1,1), padding=1))

def recon_loss_low(R, L, input_im):
    L_3 = torch.cat((L, L, L), dim=1)
    recon_loss_low = torch.mean(torch.abs(R * L_3 - input_im))
    return recon_loss_low 

def recon_loss_low_eq(R, im_eq):
    R_low_max = torch.max(R, dim=1, keepdims=True)
    recon_loss_low_eq = torch.mean(torch.abs(R_low_max[0] - im_eq))
    return recon_loss_low_eq  

def R_low_loss_smooth(R):
    R1 = torch.squeeze(R, 0)
    a=gradient(rgb_to_grayscale(R1).cuda().unsqueeze(0), "x")
    b=gradient(rgb_to_grayscale(R1).cuda().unsqueeze(0), "y")
    R_low_loss_smooth = torch.mean(torch.abs(a) + torch.abs(b))
    return R_low_loss_smooth

def smooth(I, R):
    R1 = torch.squeeze(R, 0)
    R = rgb_to_grayscale(R1)
    R = R.cuda().unsqueeze(0)
    return torch.mean(gradient(I, "x") * torch.exp(-10 * gradient(R, "x")) + gradient(I, "y") * torch.exp(-10 * gradient(R, "y")))

@LOSSES.register_module
class lowLightLoss(nn.Module):
    def __init__(self):
        super(lowLightLoss, self).__init__()
        self.recon_loss_low = recon_loss_low
        self.recon_loss_low_eq = recon_loss_low_eq
        self.R_low_loss_smooth = R_low_loss_smooth
        self.Ismooth_loss_low = smooth
        
    def forward(self, input_im, R, L):
        eq = input_im.clone()
        im_max_channel = torch.max(eq, dim=1, keepdim=True)
        im_max_channel = im_max_channel[0].squeeze(0)
        img = im_max_channel.detach().cpu().numpy()
        im_eq = histeq(img)
        im_eq = torch.from_numpy(im_eq).float().cuda()
        im_eq = im_eq.unsqueeze(0)
        recon_loss_low = self.recon_loss_low(R, L, input_im)
        recon_loss_low_eq = self.recon_loss_low_eq(R, im_eq)
        R_low_loss_smooth = self.R_low_loss_smooth(R)
        Ismooth_loss_low = self.Ismooth_loss_low(L, R)
        loss_Decom_zhangyu= recon_loss_low + 0.1 * Ismooth_loss_low + 0.1 * recon_loss_low_eq + 0.01*R_low_loss_smooth
        return loss_Decom_zhangyu


def lll():
    return lowLightLoss() 
