import torch
from skimage.util import random_noise
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
from torchsummary import summary
from LCA_LLA import LCA, LLA


class _upsample_(nn.Module) :
    def __init__(self, scale, insize, outsize) :

        super(_upsample_, self).__init__()

        # Create Layer Instance
        self._up_ = nn.Sequential(
                            nn.PixelShuffle(scale),
                            nn.Conv2d(insize, outsize, (3, 3), padding=(1, 1), bias = True),
                            )

    def forward(self, x) :
        out = self._up_(x)
        out = F.leaky_relu(out, 0.2, True)

        return out

class LAD_CNN(nn.Module):

    def __init__(self):
        super(LAD_CNN, self).__init__()
        self.conv0 = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 96, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(96, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(96, 96, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 1, kernel_size=(5, 5), padding=(2, 2))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.upsample_96 = _upsample_(2, insize=24, outsize=96)
        self.upsample_64 = _upsample_(2, insize=16, outsize=64)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.LLA_64 = LLA(d_model=64)
        self.LLA_96 = LLA(d_model=96)
        self.LCA = LCA(d_model=64)
        self.LCA_96 = LCA(d_model=96)


    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.LCA(x)
        x = self.conv3(x1)
        x = self.relu(x)
        x = self.maxpool(x)
        x2 = self.LCA_96(x)

        x = self.conv5(x2)
        x = self.relu(x)
        x = x + x2
        x = self.upsample_96(x)
        x = self.LLA_96(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x + x1
        x = self.upsample_64(x)
        x = self.LLA_64(x)
        x = self.conv6(x)
        x = x + input
        x = self.conv0(x)
        x = self.sig(x)
        return x

    def initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                # Apply Xavier Uniform Initialization
                torch.nn.init.xavier_uniform_(m.weight.data)

                if m.bias is not None :
                    m.bias.data.zero_()


if __name__ == '__main__':
    device = torch.device('cuda:0')
    writer = SummaryWriter("log_lad_cnn")

    network = LAD_CNN()
    network = network.cuda()
    print(network)
    summary(network, (1, 64, 64))