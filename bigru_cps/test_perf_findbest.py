print("Loading libraries...")

''' Imports '''
# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np

# Calculate performance
import sklearn

# Visualization imports
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom imports
from H5_dataset2 import HF_Lung_Dataset
#from models import Cnn14_8k, TwinNetworkGRU
from models_deconv import TwinNetworkGRU

import sys

''' Settings '''
# NN Hyperparameters
num_epochs = 300
batch_size = 1
learning_rate = 0.001
validation_split = .1
shuffle_dataset = False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)

threshold = 0.4

''' Data '''
print("Loading data...")

# Load data
test_dataset = HF_Lung_Dataset(train=False, dataFName='test_steth_combined.h5')

# Creating data indices for training and validation splits:
dataset_size = len(test_dataset)
indices = list(range(dataset_size))

test_sampler = SubsetRandomSampler(indices)

test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
print(f"Test: {len(indices)} samples")

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Using device: {device}")

''' Model '''
# Initialize model
print("Initializing model...")
#model = TwinNetworkGRU(64, 4, device)
model = TwinNetworkGRU(64, 4, device, gru_layers=2)
#fc_audioset_in_params = model.fc_audioset.in_features
#model.fc_audioset = nn.Linear(fc_audioset_in_params, 6, bias=True)

exp_name = sys.argv[1]

checkpoint = torch.load(f"models/model_{exp_name}.pt")

# Remove all the "module." from the keys because that's there for some reason
checkpoint_clean = {}
for i in checkpoint.keys():
    checkpoint_clean[i.replace("module.", "")] = checkpoint[i]

model.load_state_dict(checkpoint_clean)

model.to(device)

''' Testing '''
# Train the model
ranks = []
print("Testing model...")
with torch.no_grad():
    total = 0
    correct = 0
    acc_total = 0

    for specs, labels, files in tqdm(test_loader):
        if labels[0].sum(axis=0)[3] == 0:# and labels[0].sum(axis=0)[2] == 0:
            continue
        scores = np.zeros(4)
        inp = specs.float().to(device)
        outputs, _ = model(inp)

        batch_size = labels.shape[0]
        total += batch_size

        predicted = outputs > threshold
        
        #print(predicted.shape)

        #print(predicted, labels)
        #same = predicted == labels.to(device)
        #same = same.sum(axis=1).sum(axis=0).cpu().numpy()
        #correct = np.add(correct, same)

        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            correct = same.all(axis=1).sum()
        acc_total += labels.shape[0] * labels.shape[1]

        # calculate F1 score for each of six classes
        predicted = predicted.reshape(
            (labels.shape[0] * labels.shape[1], 4)
        ).transpose(1, 0)
        labels = labels.reshape(
            (labels.shape[0] * labels.shape[1], 4)
        ).transpose(1, 0)
        #print(labels.shape)
        '''
        for i in range(4):
            scores[i] = batch_size * sklearn.metrics.f1_score(
                predicted[i].cpu().numpy(),
                labels[i].cpu().numpy(),
                zero_division=0
            )
        '''
        ranks.append([files[0], correct / (labels.shape[1])])
        #print(files[0], correct / acc_total)
        #if correct / (labels.shape[0] * labels.shape[1]) > 0.8:
        #    break
ranks.sort(key=lambda x: x[1], reverse=True)
print(ranks[:10])