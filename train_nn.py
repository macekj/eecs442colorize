import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# Use GPU to train
device = torch.device('cuda:0')

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
num_epoch = 20


# model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(model, train_loader, val_loader, num_epoch):
    trn_loss_hist = []
    trn_acc_hist = []
    val_acc_hist = []
    model.train()
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
        trn_acc_hist.append(evaluate(model, train_loader))
        print("\n Evaluate on validation set...")
        val_acc_hist.append(evaluate(model, val_loader))
    print('Done training')
    return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for img_batch, true_img in tqdm(loader):
            img_batch = img_batch.to(device)
            true_img = true_img.to(device)
            output = model(img_batch)
            psnr = calc_psnr(output, true_img)
            accuracies.append(psnr)
        avg_score = np.mean(accuracies)
        print("\n Evaluation accuracy: {}".format(avg_score))
        return avg_score


def calc_psnr(pred_img, real_img):
    return 0
