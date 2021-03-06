import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from math import sqrt

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, 0.0, 1 / sqrt(3 * 3 * m.weight.size(1)))
        # nn.init.constant_(m.bias, 0.0)
    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 0.0, 1 / sqrt(m.weight.size(0)))

class ColorNet(BaseColor):
  def __init__(self):
    super(ColorNet, self).__init__()


    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
        # nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
    )

    self.conv5 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
    )

    self.conv6 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
    )

    self.conv7 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
    )

    self.conv8 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
    )

    self.softmax = nn.Softmax(dim=1)
    self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    print('initializing nn weights')
    # Initialize neural network weights
    for conv in [self.conv1, self.conv2, self.conv3, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
        conv.apply(init_weights)

    init_weights(self.model_out)
    print('weights initialized')


  def forward(self, x):
    # x.shape is [N, 1, 128, 128]
    # and contains the L layer

    # normalize luminances as they go into forward pass
    out = self.normalize_l(x)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.conv7(out)
    out = self.conv8(out)
    out = self.softmax(out)
    out = self.model_out(out)
    out = self.upsample(out)
    # unnormalize the AB layers as they come out
    out = self.unnormalize_ab(out)
    
    # at this point, out.shape is [N, 2, 128, 128]
    # and contains the AB layers

    return out

