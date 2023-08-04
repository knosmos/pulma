print("Loading libraries...")

''' Imports '''
# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn

# Visualization imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import os

# Custom imports
from H5_dataset import HF_Lung_Dataset
import people_split
#from models import TwinNetwork, Cnn14_8k
#import librosa
#import librosa.display

import time
import sys
import argparse

from copy import deepcopy

''' Settings '''
# Argparser
parser = argparse.ArgumentParser(description='Train PANN + CPS on the HF Lung dataset')

# Experiment/Data
parser.add_argument('--name', type=str, default="exp", help='Name of the experiment')
parser.add_argument('--percent', type=float, default=25, help='Percentage of supervised data to use')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint to resume training from')

parser.add_argument('--seed', type=int, default=2, help='Random seed')

# NN Hyperparameters
num_epochs = 200
batch_size = 64
learning_rate = 0.001 #0.01
validation_split = .1
shuffle_dataset = False # Prevent train/val mixing

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train for')
parser.add_argument('--batch', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--val', type=float, default=0.1, help='Validation split')

# Get arguments
args = parser.parse_args()
experiment_name = args.name
percent_data = args.percent
model_checkpoint = args.checkpoint

num_epochs = args.epochs
batch_size = args.batch
learning_rate = args.lr
validation_split = args.val

# Other settings
random_seed = args.seed # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)
np.random.seed(random_seed)

threshold = 0.5

''' Data '''
print("Loading data...")

# Load data
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_steth_combined.h5')

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
split = int(np.floor(validation_split * dataset_size))

tot = np.array([1, 1.427409854, 2.177351724, 1.951449024]) # supposed weights for entire dataset
best = 1000
best_seed = -1
for seed in tqdm(range(200)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    people = deepcopy(people_split.people) 
    np.random.shuffle(people)
    #print("number of people:", len(people))
    indices = []
    for person in people:
        indices.extend(person)
    train_indices, val_indices = indices[split:], indices[:split]

    diff = 0
    for percent_data in [6.25, 12.5, 25, 50]:
        # extract percentage of data
        train_labeled_indices = train_indices[:int((len(train_indices) * percent_data) // 100)]
        train_unlabeled_indices = train_indices[int((len(train_indices) * percent_data) // 100):]

        # Creating PT data samplers and loaders:
        train_labeled_sampler = SubsetRandomSampler(train_labeled_indices)
        train_unlabeled_sampler = SubsetRandomSampler(train_unlabeled_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_labeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_labeled_sampler)
        train_unlabeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_unlabeled_sampler)
        validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        #print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

        # get stats of dataset
        wt = np.zeros(4)
        for specs, label_batch in train_labeled_loader:
            wt = np.add(wt, label_batch.sum(dim=1).sum(dim=0))
        m = int(wt[0])
        for i in range(4):
            wt[i] = m / wt[i]
        #print(f"{percent_data}%: {wt}")
        diff += np.sum(np.abs(wt.cpu().numpy() - tot)) ** 2
    if diff < best and seed != 50:
        best = diff
        best_seed = seed

print(f"best seed: {best_seed}")
np.random.seed(best_seed)
torch.manual_seed(best_seed)

people = deepcopy(people_split.people) 
np.random.shuffle(people)
print("number of people:", len(people))
indices = []
for person in people:
    indices.extend(person)
train_indices, val_indices = indices[split:], indices[:split]

for percent_data in [6.25, 12.5, 25, 50]:
    train_labeled_indices = train_indices[:int((len(train_indices) * percent_data) // 100)]
    train_unlabeled_indices = train_indices[int((len(train_indices) * percent_data) // 100):]

    # Creating PT data samplers and loaders:
    train_labeled_sampler = SubsetRandomSampler(train_labeled_indices)
    train_unlabeled_sampler = SubsetRandomSampler(train_unlabeled_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_labeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_labeled_sampler)
    train_unlabeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_unlabeled_sampler)
    validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

    #print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

    # get stats of dataset
    wt = np.zeros(4)
    for specs, label_batch in train_labeled_loader:
        wt = np.add(wt, label_batch.sum(dim=1).sum(dim=0))
    m = int(wt[0])
    for i in range(4):
        wt[i] = m / wt[i]
    print(f"{percent_data}%: {wt}")

'''
numofeach = np.zeros(4)
for specs, label_batch in train_unlabeled_loader:
    for labels in label_batch:
        for label in labels:
            numofeach = np.add(numofeach, label)
print("train, unlabeled:")
print(numofeach.cpu().numpy().astype(np.int64))

numofeach = np.zeros(4)
for specs, label_batch in validation_loader:
    for labels in label_batch:
        for label in labels:
            numofeach = np.add(numofeach, label)
print("validation:")
print(numofeach.cpu().numpy().astype(np.int64))
'''