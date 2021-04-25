import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ColorNet(nn.Module):
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


  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.softmax(x)
    x = self.model_out(x)
    x = self.upsample(x)

    return x

