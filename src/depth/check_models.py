from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F

from layers import *
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder

models = {}
parameters_to_train = []

device = torch.device("cuda")

num_scales = 4
num_input_frames = 3  # one sample of 3 frames
num_pose_frames = 2

models["encoder"] = ResnetEncoder(
    18, False)
models["encoder"].to(device)
parameters_to_train += list(models["encoder"].parameters())

models["depth"] = DepthDecoder(
    models["encoder"].num_ch_enc, [0, 1, 2, 3])
models["depth"].to(device)
parameters_to_train += list(models["depth"].parameters())

# print(models['encoder'])

imgs = torch.randn(4, 3, 256, 960).cuda()
encoder_features = models['encoder'](imgs)
# print(encoder_features)

for idx in encoder_features:
    print(idx.shape)

# print(models['depth'])

res = models['depth'](encoder_features)
for itm in res:
    print(itm, res[itm].shape)