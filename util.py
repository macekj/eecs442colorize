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

class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()

    def normalize_l(self, img):
        return (img - 50.0)/100.0

    def unnormalize_l(self, img):
        return (img*100.0) + 50.0

    def normalize_ab(self, img):
        return img / 110.0

    def unnormalize_ab(self, img):
        return img * 110.0


def get_results_combined(testloader, net, device, folder='output_train'):
  result = []
  cnt = 1
  os.makedirs(folder, exist_ok=True)
  with torch.no_grad():
    net = net.eval()
    cnt = 0
    for img, true_ab in tqdm(testloader):
      img = img.to(device)
      true_ab = true_ab.to(device)
      output = net(img)[0].cpu().numpy()
      output = np.concatenate((img[0].cpu().numpy(), output), 0)
      output = np.moveaxis(output, 0, -1)
      input_img = np.moveaxis(img[0].cpu().numpy(), 0, -1)
      c, h, w = output.shape
      rgb = color.lab2rgb(output)

      plt.subplot(1, 2, 1)
      plt.imshow(input_img, cmap='gray')
      plt.title('Input Image', fontsize=32)
      plt.axis('off')

      plt.subplot(1, 2, 2)
      plt.imshow(rgb)
      plt.title('Colorized Output', fontsize=32)
      plt.axis('off')


      plt.gcf().set_size_inches(18, 10)
      plt.savefig('./{}/out{}.png'.format(folder, cnt))
      cnt += 1

def get_result(testloader, net, device, folder='output_train'):
  result = []
  cnt = 1
  os.makedirs(folder, exist_ok=True)
  with torch.no_grad():
    net = net.eval()
    cnt = 0
    for img, true_ab in tqdm(testloader):
      img = img.to(device)
      true_ab = true_ab.to(device)
      output = net(img)[0].cpu().numpy()
      output = np.concatenate((img[0].cpu().numpy(), output), 0)
      output = np.moveaxis(output, 0, -1)
      c, h, w = output.shape
      rgb = color.lab2rgb(output)
      print("Saving images x,l,a,b{}.png".format(cnt))
      plt.imsave('./{}/x{}.png'.format(folder, cnt), rgb)
      plt.imsave('./{}/l{}.png'.format(folder, cnt), output[:, :, 0])
      plt.imsave('./{}/a{}.png'.format(folder, cnt), output[:, :, 1])
      plt.imsave('./{}/b{}.png'.format(folder, cnt), output[:, :, 2])
      cnt += 1
