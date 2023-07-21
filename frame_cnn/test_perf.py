print("Loading libraries...")

''' Imports '''
# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np

# Visualization imports
from tqdm import tqdm

# Custom imports
from custom_dataset import HF_Lung_Dataset
from model_old import FrameNet

''' Settings '''
# NN Hyperparameters
num_epochs = 300
batch_size = 512
learning_rate = 0.001
validation_split = .3
shuffle_dataset = False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)

threshold = 0.5

''' Data '''
print("Loading data...")

# Load data
test_dataset = HF_Lung_Dataset(train=False)

# Creating data indices for training and validation splits:
dataset_size = len(test_dataset)
indices = list(range(dataset_size))

test_sampler = SubsetRandomSampler(indices)

test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
print(f"Test: {len(indices)} samples")

''' Model '''
# Initialize model
print("Initializing model...")
model = FrameNet(64)
model.load_state_dict(torch.load("model_old.pt"))

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

''' Testing '''
# Train the model
print("Testing model...")
with torch.no_grad():
    correct = 0
    total = 0
    for specs, labels in tqdm(test_loader):
        outputs = model(specs.float().to(device)).to(device)
        total += labels.size(0)
        predicted = outputs > threshold
        #print(predicted, labels)
        correct += (predicted == labels.to(device)).all(axis=1).sum().item()

    print(f'Test accuracy: {(correct / total) * 100:.2f}%')