import sys
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torchsummary import summary
from cnn import ColorNet
from dataset import *
from util import *

# This script loads a pre-trained model and saves combined images
# of the input data and predicted colorization

if len(sys.argv) < 5:
    print("USAGE: python3 test.py state_dict_file src_folder dest_folder num_generate")
    exit(1)

# Use GPU to train
device = torch.device('cuda:0')
# Uncomment to use CPU instead
# device = torch.device('cpu')

name = 'colorization_net'
model = ColorNet().to(device)
model.load_state_dict(torch.load(sys.argv[1]))

test_range = (1, int(sys.argv[4]))
test_data = ColorizeDataset(flag='test', dataDir=sys.argv[2], data_range=test_range)
test_loader = DataLoader(test_data, batch_size=1)

get_results_combined(test_loader, model, device, folder=sys.argv[3])