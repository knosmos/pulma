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

# Visualization imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import os

# Custom imports
from custom_dataset import HF_Lung_Dataset
from model_old import FrameNet

''' Settings '''
# NN Hyperparameters
num_epochs = 50
batch_size = 2048 #512
learning_rate = 0.0005 #0.001
validation_split = .25
shuffle_dataset = False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)

threshold = 0.5

''' Data '''
print("Loading data...")

# Load data
train_dataset = HF_Lung_Dataset(train=True)

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

# Result storage
csv = open(f"results/res_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")

''' Model '''
# Initialize model
print("Initializing model...")
model = FrameNet(64)

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Loss and optimizer
#criterion = nn.BCEWithLogitsLoss() # Needed since we are doing multilabel classification
criterion = nn.BCELoss() # Needed since we are doing multilabel classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

''' Training '''
# Train the model
print("Training model...")
total_step = len(train_loader)

best_val_acc = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for i, (specs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(specs.float().to(device)).to(device)
        loss = criterion(outputs, labels.to(device)).float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        predicted = outputs > threshold
        correct = (predicted == labels.to(device)).all(axis=1).sum().item()
        #print(f'predicted={predicted}, labels={labels}, correct = {correct}')

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct / total * 100 :.2f}%")

        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_total += total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / 128:.4f}, Accuracy: {epoch_correct / epoch_total * 100 :.2f}%")

    ''' Validation '''
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for specs, labels in validation_loader:
            outputs = model(specs.float().to(device)).to(device)
            total += labels.size(0)
            predicted = outputs > threshold
            #print(predicted, labels)
            correct += (predicted == labels.to(device)).all(axis=1).sum().item()

        print(f'Validation accuracy: {(correct / total) * 100:.2f}%')
        
        # write to csv
        csv.write(f"{epoch+1},{epoch_loss / 128},{epoch_correct / epoch_total * 100},{correct / total * 100}\n")
        csv.flush()
        os.fsync(csv.fileno())

        # Save the model checkpoint if validation accuracy improves
        if (correct / total) > best_val_acc:
            print('Saving model...')
            torch.save(model.state_dict(), 'model_old.pt')
            best_val_acc = correct / total
