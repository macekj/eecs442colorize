import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from PIL import Image
import torchvision
from colormap.colors import Color, hex2rgb
from skimage import color
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

import pdb

class ColorizeDataset(Dataset):
  def __init__(self, flag, dataDir='./colorize_dataset/', data_range=(0, 8)):
    assert(flag in ['train', 'test'])
    print("load "+ flag+" dataset start")
    print("    from: %s" % dataDir)
    print("    range: [%d, %d)" % (data_range[0], data_range[1]))
    self.dataset = []
    self.cluster = MiniBatchKMeans(n_clusters=313)
    self.fit_cluster()
    for i in range(data_range[0], data_range[1]):
      img_filename = 'gry_%05d.jpg' % i
      exp_filename = 'clr_%05d.jpg' % i

      # Originally, I was just using the B&W images I generated from the training set.
      # Looks like PIL uses a different LAB calculation than skimage, so simply
      # re-pull the L layer as the input rather than use the one on disk
      # Normalize input image
      # img = Image.open(os.path.join(dataDir,flag,img_filename))
      # img = np.asarray(img).astype("f") / 255.0
      # img = np.expand_dims(img, axis=0)

      # Normalize expected image and convert to LAB
      exp = Image.open(os.path.join(dataDir,flag,exp_filename))
      exp = np.asarray(exp).astype("f") / 255.0 # TODO: normalize?

      # Skip images with an expected B&W output
      if exp.ndim != 3 or exp.shape[2] != 3:
        continue

      exp_lab = np.moveaxis(color.rgb2lab(exp), -1, 0)
      img = exp_lab[0:1, :, :]

      exp_ab = exp_lab[1:, :, :]
      
      #exp_ab = exp_lab[1:, :, :]


      self.dataset.append((img, exp_ab))
    print("load dataset done")

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    img, exp_ab = self.dataset[index]
    return torch.FloatTensor(img), torch.FloatTensor(exp_ab)

  def fit_cluster(self):
    # Open image corresponding to full color space and normalize
    color_space = Image.open(os.path.join("color_space.jpg"))
    color_space = np.asarray(color_space).astype("f")/255.0

    # Convert to L*a*b* and remove L channel
    color_space = cv2.cvtColor(color_space, cv2.COLOR_RGB2LAB)
    ABchannel = color_space[:,:,1:3]

    # Transfrom to feature vector and fit k means clustering
    ABchannel = ABchannel.reshape((color_space.shape[0]*color_space.shape[1], 2))
    self.cluster.fit(ABchannel)

