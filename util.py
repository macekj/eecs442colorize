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
from dataset import cluster
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


# As of now, all functions pasted below here are just from HW5 starter code for convenience

# def save_label(label, path):
#   '''
#   Function for ploting labels.
#   '''
#   colormap = [
#       '#000000',
#       '#0080FF',
#       '#80FF80',
#       '#FF8000',
#       '#FF0000',
#   ]
#   assert(np.max(label)<len(colormap))
#   colors = [hex2rgb(color, normalise=False) for color in colormap]
#   w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
#   with open(path, 'wb') as f:
#       w.write(f, label)

# def train(trainloader, net, criterion, optimizer, device, epoch):
#   '''
#   Function for training.
#   '''
#   start = time.time()
#   running_loss = 0.0
#   cnt = 0
#   net = net.train()
#   for images, labels in tqdm(trainloader):
#     images = images.to(device)
#     labels = labels.to(device)
#     optimizer.zero_grad()
#     output = net(images)
#     loss = criterion(output, labels)
#     loss.backward()
#     optimizer.step()
#     running_loss += loss.item()
#     cnt += 1
#   end = time.time()
#   running_loss /= cnt
#   print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
#         (epoch, running_loss, end-start))
#   return running_loss

# def test(testloader, net, criterion, device):
#   '''
#   Function for testing.
#   '''
#   losses = 0.
#   cnt = 0
#   with torch.no_grad():
#     net = net.eval()
#     for images, labels in tqdm(testloader):
#       images = images.to(device)
#       labels = labels.to(device)
#       output = net(images)
#       loss = criterion(output, labels)
#       losses += loss.item()
#       cnt += 1
#   print('\n',losses / cnt)
#   return (losses/cnt)


# def cal_AP(testloader, net, criterion, device):
#   '''
#   Calculate Average Precision
#   '''
#   losses = 0.
#   cnt = 0
#   with torch.no_grad():
#     net = net.eval()
#     preds = [[] for _ in range(5)]
#     heatmaps = [[] for _ in range(5)]
#     for images, labels in tqdm(testloader):
#       images = images.to(device)
#       labels = labels.to(device)
#       output = net(images).cpu().numpy()
#       for c in range(5):
#         preds[c].append(output[:, c].reshape(-1))
#         heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

#     aps = []
#     for c in range(5):
#       preds[c] = np.concatenate(preds[c])
#       heatmaps[c] = np.concatenate(heatmaps[c])
#       if heatmaps[c].max() == 0:
#         ap = float('nan')
#       else:
#         ap = ap_score(heatmaps[c], preds[c])
#         aps.append(ap)
#       print("AP = {}".format(ap))
#     print("Average Precision (all classes) = {}".format(np.mean(aps)))
#   return None


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
      c, h, w = output.shape
      
      # Convert color probabilities to labels and reshape to vector
      output = prob_to_color_labels(output, 0.35)
      output = output.reshape((output.shape[0]*output.shape[1]))
      output = cluster.cluster_centers_.astype("f")[output]
      output = output.reshape((h,w,2))
      Llayer = np.moveaxis(img[0].cpu().numpy(), 0, -1)
      output = np.concatenate((output, Llayer), 2)
      #output = np.concatenate((img[0].cpu().numpy(), output), 0)
      #output = np.moveaxis(output, 0, -1)
      c, h, w = output.shape
      # y = np.argmax(output, 0).astype('uint8')
      # gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
      # save_label(y, './{}/y{}.png'.format(folder, cnt))
      # save_label(gt, './{}/gt{}.png'.format(folder, cnt))
      # plt.imsave('./{}/x{}.png'.format(folder, cnt),
      #            ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))
      rgb = color.lab2rgb(output)
      print("Saving images x,l,a,b{}.png".format(cnt))
      plt.imsave('./{}/x{}.png'.format(folder, cnt), rgb)
      plt.imsave('./{}/l{}.png'.format(folder, cnt), output[:, :, 0])
      plt.imsave('./{}/a{}.png'.format(folder, cnt), output[:, :, 1])
      plt.imsave('./{}/b{}.png'.format(folder, cnt), output[:, :, 2])
      cnt += 1

def prob_to_color_labels(Z, temp):
  Z = np.exp(np.log(Z)/temp)
  Z = Z / np.sum(Z)
  Z = np.mean(Z, axis=0).astype("uint8")
  return Z


# def plot_hist(trn_hist, val_hist):
#     x = np.arange(len(trn_hist))
#     plt.figure()
#     plt.plot(x, trn_hist)
#     plt.plot(x, val_hist)
#     plt.legend(['Training', 'Validation'])
#     plt.xticks(x)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()