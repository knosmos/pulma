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
from H5_dataset2 import HF_Lung_Dataset
from models import TwinNetwork, Cnn14_8k
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
parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle dataset')

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
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_steth2.h5')

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
split = int(np.floor(validation_split * dataset_size))
people = people_split.people
np.random.shuffle(people)
print("number of people:", len(people))
indices = []
for person in people:
    indices.extend(person)

# Creating dataloader
#loader = DataLoader(train_dataset, batch_size=1)

print(f"Loaded: {len(indices)} samples")

''' Model '''
# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# get stats of dataset
numofeach = []
for idx in range(len(train_dataset)):
    specs, labels, filename = train_dataset[indices[idx]]
    labels = labels.sum(dim=0)
    if labels[0] > 1000:
        print(idx, filename)
    numofeach.append(labels.cpu().numpy().astype(np.int64))

# plot

numofeach = np.array(numofeach)
numofeach = numofeach.transpose()
print(numofeach.shape)

plt.figure()
plt.plot(numofeach[5])

#plt.legend(["I", "E", "D", "Rhonchi", "Wheeze", "Stridor"])
plt.title("Distribution of stridor in shuffled dataset")
#plt.savefig(f"results/{experiment_name}_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.png")
plt.close()