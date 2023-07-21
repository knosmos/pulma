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

# Custom imports
from H5_dataset import HF_Lung_Dataset
from model_simple import Baseline

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

threshold = 0.4

''' Data '''
print("Loading data...")

# Load data
test_dataset = HF_Lung_Dataset(train=False, dataFName='test_steth.h5')

# Creating data indices for training and validation splits:
dataset_size = len(test_dataset)
indices = list(range(dataset_size))

test_sampler = SubsetRandomSampler(indices)

test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
print(f"Test: {len(indices)} samples")

''' Model '''
# Initialize model
print("Initializing model...")
model = Baseline(length=938, n_mels=128, num_classes=6)
model.load_state_dict(torch.load("model.pt"))

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

''' Testing '''
# Train the model
print("Testing model...")
with torch.no_grad():
    scores = np.zeros(6)
    bscores = np.zeros(6)
    total = 0
    for specs, labels in tqdm(test_loader):
        outputs = model(specs.float().to(device)).to(device)
        batch_size = labels.shape[0]
        total += batch_size
        predicted = outputs > threshold
        
        #print(predicted.shape)

        #print(predicted, labels)
        #same = predicted == labels.to(device)
        #same = same.sum(axis=1).sum(axis=0).cpu().numpy()
        #correct = np.add(correct, same)

        # calculate F1 score for each of six classes
        predicted = predicted.reshape(
            (labels.shape[0] * labels.shape[1], 6)
        ).transpose(1, 0)
        labels = labels.reshape(
            (labels.shape[0] * labels.shape[1], 6)
        ).transpose(1, 0)
        #print(labels.shape)
        for i in range(6):
            scores[i] += batch_size * sklearn.metrics.f1_score(
                predicted[i].cpu().numpy(),
                labels[i].cpu().numpy()
            )
            bscores[i] += batch_size * sklearn.metrics.fbeta_score(
                predicted[i].cpu().numpy(),
                labels[i].cpu().numpy(),
                beta=2
            )

    print(f'Test F1 Score: {(scores / total) * 100}%')
    print(f'Test F2 Score: {(bscores / total) * 100}%')