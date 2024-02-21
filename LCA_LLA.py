import torch
from skimage.util import random_noise
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
from torchsummary import summary

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1))


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LCA(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1))
        self.activation_gelu = nn.GELU()
        self.proj_2 = nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1))
        self.proj_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model, (1, 1), bias=True)
        )

    def forward(self, input_x):
        x = self.proj_1(input_x)
        x1 = F.leaky_relu(x, 0.2, True)
        y = self.proj_3(x1)
        y = x1*y
        y = self.proj_2(y)


        output = input_x + y

        return output

class LLA(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)

        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = x + shorcut
        x = self.proj_2(x)
        return x
