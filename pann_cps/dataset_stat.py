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
from torch.utils.tensorboard import SummaryWriter
import sklearn

# Visualization imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import os

# Custom imports
from H5_dataset import HF_Lung_Dataset
from models import TwinNetwork, Cnn14_8k
import librosa
import librosa.display

import people_split

import time
import sys
import argparse

''' Settings '''
# Argparser
parser = argparse.ArgumentParser(description='Train PANN + CPS on the HF Lung dataset')

# Experiment/Data
parser.add_argument('--name', type=str, default="exp", help='Name of the experiment')
parser.add_argument('--percent', type=int, default=25, help='Percentage of supervised data to use')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint to resume training from')

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
parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset')

# Get arguments
args = parser.parse_args()
experiment_name = args.name
percent_data = args.percent
model_checkpoint = args.checkpoint

num_epochs = args.epochs
batch_size = args.batch
learning_rate = args.lr
validation_split = args.val
shuffle_dataset = args.shuffle

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)
np.random.seed(random_seed)

threshold = 0.5

''' Data '''
print("Loading data...")

# Load data
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_steth.h5')

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
split = int(np.floor(validation_split * dataset_size))
people = people_split.people
np.random.shuffle(people)
print("number of people:", len(people))
indices = []
for person in people:
    indices.extend(person)
train_indices, val_indices = indices[split:], indices[:split]
# extract percentage of data
train_labeled_indices = train_indices[:(len(train_indices) * percent_data) // 100]
train_unlabeled_indices = train_indices[(len(train_indices) * percent_data) // 100:]

# Creating PT data samplers and loaders:
train_labeled_sampler = SubsetRandomSampler(train_labeled_indices)
train_unlabeled_sampler = SubsetRandomSampler(train_unlabeled_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_labeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_labeled_sampler)
train_unlabeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_unlabeled_sampler)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

# Result storage
csv = open(f"results/{experiment_name}_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")
writer = SummaryWriter()

''' Model '''
# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

numofeach = np.zeros(6)
for specs, label_batch in validation_loader:
    for labels in label_batch:
        for label in labels:
            numofeach = np.add(numofeach, label)
print("validation:")
print(numofeach.cpu().numpy().astype(np.float64)/(len(val_indices) * 1500))
print(numofeach.cpu().numpy().astype(np.float64))

# get stats of dataset
numofeach = np.zeros(6)
for specs, label_batch in train_labeled_loader:
    for labels in label_batch:
        for label in labels:
            numofeach = np.add(numofeach, label)
print("train, labeled:")
print(numofeach.cpu().numpy().astype(np.float64)/(len(train_labeled_indices) * 1500))
print(numofeach.cpu().numpy().astype(np.float64))

numofeach = np.zeros(6)
for specs, label_batch in train_unlabeled_loader:
    for labels in label_batch:
        for label in labels:
            numofeach = np.add(numofeach, label)
print("train, unlabeled:")
print(numofeach.cpu().numpy().astype(np.float64)/(len(train_unlabeled_indices) * 1500))
print(numofeach.cpu().numpy().astype(np.float64))