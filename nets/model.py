import time
import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import cv2
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward, DWTInverse
from utils.tool_func import *


class dilated_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x

    def forward(self, x):
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')

        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResUNet34_StyleAdvSG(nn.Module):
    def __init__(self, out_c=2, pretrained=True, fixed_feature=False):
        super().__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False

        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)

        self.ce = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=1)
        self.conv11 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(64)

        self.block1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.block2 = nn.Sequential(
            self.resnet.maxpool,
            self.resnet.layer1
        )

        self.block3 = nn.Sequential(
            self.resnet.layer2
        )

        self.block4 = nn.Sequential(
            self.resnet.layer3
        )

        self.block5 = nn.Sequential(
            self.resnet.layer4
        )
        self.decoder_G = Decoder_imageS(channelList=[64, 64, 128, 256, 512])
        self.decoder_L = Decoder_imageS(channelList=[64, 64, 128, 256, 512])

    def forward(self, x, style_mean=None, style_std=None):
        x = c1 = self.block1(x)
        x = c2 = self.block2(x)
        x = c3 = self.block3(x)
        x = c4 = self.block4(x)
        x = content_feat =  self.block5(x)

        xStyle = x
        if (style_mean is not None):
            style_mean = style_mean.detach()
            style_std = style_std.detach()

            size = content_feat.size()
            content_mean, content_std = calc_mean_std(content_feat)
            normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
            xStyle = normalized_feat * style_std.expand(size) + style_mean.expand(size)

        x_G = self.decoder_G(xStyle, c1, c2, c3, c4)
        x_L = self.decoder_L(xStyle, c1, c2, c3, c4)

        return x_L, [x_G, xStyle]

    def forward_block1(self, x):
        x = c1 = self.block1(x)

        return x, c1


    def forward_block2(self, x, c1):
        x = c2 = self.block2(x)

        return x, c1, c2


    def forward_block3(self, x, c1, c2):
        x = c3 = self.block3(x)

        return x, c1, c2, c3


    def forward_block4(self, x, c1, c2, c3):
        x = c4 = self.block4(x)

        return x, c1, c2, c3, c4

    def forward_block5(self, x, c1, c2, c3, c4):
        x = self.block5(x)

        return x, c1, c2, c3, c4

    def forward_rest_G(self, x, c1, c2, c3, c4):
        out = self.decoder_G(x, c1, c2, c3, c4)

        return out

    def forward_rest_L(self, x, c1, c2, c3, c4):
        out = self.decoder_L(x, c1, c2, c3, c4)

        return out
