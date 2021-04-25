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

import pdb

class ColorizeDataset(Dataset):
  def __init__(self, flag, dataDir='./colorize_dataset/', data_range=(0, 8)):
    assert(flag in ['train', 'test'])
    print("load "+ flag+" dataset start")
    print("    from: %s" % dataDir)
    print("    range: [%d, %d)" % (data_range[0], data_range[1]))
    self.dataset = []
    for i in range(data_range[0], data_range[1]):
      img_filename = 'gry_%05d.jpg' % i
      exp_filename = 'clr_%05d.jpg' % i

      # Normalize input image
      img = Image.open(os.path.join(dataDir,flag,img_filename))
      img = np.asarray(img).astype("f")/128.0-1.0
      img = np.expand_dims(img, axis=0)

      # Normalize expected image and convert to LAB
      exp = Image.open(os.path.join(dataDir,flag,exp_filename))
      exp = np.asarray(exp).astype("f") # TODO: normalize?

      # Skip images with an expected B&W output
      if exp.ndim != 3 or exp.shape[2] != 3:
        continue

      exp_lab = color.rgb2lab(exp)

      self.dataset.append((img, exp_lab))
    print("load dataset done")

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    img, label = self.dataset[index]
    return torch.FloatTensor(img), torch.FloatTensor(label)

# train_range = (0, 8000)
# val_range = (8000, 10000)
# test_range = (1, 2000)

train_range = (0, 800)
val_range = (800, 1000)
test_range = (1, 200)


train_data = ColorizeDataset(flag='train', data_range=train_range)
train_loader = DataLoader(train_data, batch_size=4)

val_data = ColorizeDataset(flag='train', data_range=val_range)
val_loader = DataLoader(val_data, batch_size=4)

test_data = ColorizeDataset(flag='test', data_range=test_range)
test_loader = DataLoader(test_data, batch_size=1)

