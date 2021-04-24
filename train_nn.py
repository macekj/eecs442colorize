import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn


# Use GPU to train
device = torch.device('cuda:0')

max_pixel_val = torch.tensor(127)  # AB channels have expected max value of 127


# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
num_epoch = 20


# model = Network().to(device)
criterion = nn.CrossEntropyLoss()  # TODO: Replace with Zhang loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(model, train_loader, val_loader, num_epoch):
    model.train()
    trn_loss_hist = []
    val_loss_hist = []
    print('Beginning training')
    for i in range(num_epoch):
        running_loss = []
        for img_batch, true_img in tqdm(train_loader):
            img_batch = img_batch.to(device)
            true_img = true_img.to(device)
            optimizer.zero_grad()
            output = model(img_batch)
            loss = criterion(output, true_img)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))
        trn_loss_hist.append(np.mean(running_loss))
        val_loss = test(model, val_loader)
        print("\n Val Loss:{}".format(val_loss))
        val_loss_hist.append(val_loss)
    print('Done training')
    return trn_loss_hist, val_loss_hist


def test(model, test_loader):
    model.eval()
    running_loss = []
    with torch.no_grad():
        for img_batch, true_img in tqdm(test_loader):
            img_batch = img_batch.to(device)
            true_img = true_img.to(device)
            output = model(img_batch)
            loss = criterion(output, true_img)
            running_loss.append(loss.item())
    return np.mean(running_loss)


# Expects a batch size of 1 for the test data
def evaluate_model(model, loader):
    model.eval()
    psnr_vals = []
    with torch.no_grad():
        for pred_img, true_img in tqdm(loader):
            pred_img = pred_img.to(device)
            true_img = true_img.to(device)
            output = model(pred_img)
            psnr = calc_psnr(output, true_img)
            psnr_vals.append(psnr)
        avg_psnr = np.mean(psnr_vals)
        print("\n Avg PSNR: {}".format(avg_psnr))
        return avg_psnr


def calc_psnr(pred_img, real_img):
    # Calculate MSE
    sq_diff = torch.square(real_img - pred_img)
    mse = (1 / (real_img.shape[0] * real_img.shape[1] * real_img.shape[2])) * torch.sum(sq_diff)
    return 20 * torch.log10(max_pixel_val) - 10 * torch.log10(mse)
