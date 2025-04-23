import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from collections import OrderedDict

from torch import nn
import torch.nn.functional as F

from .depth_decoder import DepthDecoder
from .resnet_encoder import ResnetEncoder

from src.model_dep import resnet50, Up

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()

        self.res_encoder = ResnetEncoder(34, True)
        self.depth_decoder = DepthDecoder(self.res_encoder.num_ch_enc, [0, 1, 2, 3])

    def forward(self, x):
        feats = self.res_encoder(x)
        out = self.depth_decoder(feats)  # [('disp', 0)]

        return out

class DepthEncode(nn.Module):
    def __init__(self):
        super(DepthEncode, self).__init__()

        trunk = resnet50(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up1 = Up(1024 + 512, 256, scale_factor=(2, 2))
        self.up2 = Up(256 + 256, 256, scale_factor=(2, 2))
        self.head_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 71, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B, 64, 128, 480

        x1 = self.layer1(x)  # B, 256, 128, 480
        x2 = self.layer2(x1)  # B, 512, 64, 240
        x = self.layer3(x2)  # B, 1024, 32, 120

        x = self.up1(x, x2)
        x = self.up2(x, x1)

        x = self.head_conv(x)

        return x