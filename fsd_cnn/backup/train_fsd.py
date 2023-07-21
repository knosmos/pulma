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
from models import Cnn14_DecisionLevelMax
import librosa
import librosa.display

import time

# Tensorboard
writer = SummaryWriter()

''' Settings '''
# NN Hyperparameters
num_epochs = 200
batch_size = 1024
learning_rate = 0.001 #0.01
validation_split = .1
shuffle_dataset = False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)
np.random.seed(random_seed)

threshold = 0.5

''' Data '''
print("Loading data...")

# Load data
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_fsd.h5')
val_dataset = HF_Lung_Dataset(train=True, dataFName='val_fsd.h5')

# Creating data indices for training and validation splits:
'''
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
'''

train_indices = list(range(len(train_dataset)))
val_indices = list(range(len(val_dataset)))

if shuffle_dataset:
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

# Result storage
csv = open(f"results/fsd_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")

''' Model '''
# Initialize model
print("Initializing model...")
model = Cnn14_DecisionLevelMax(sample_rate=4000, window_size=256, hop_size=80, mel_bins=64, fmin=50, fmax=14000, classes_num=200)

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model.to(device)
#model = torch.nn.DataParallel(model)

# Loss and optimizer
#criterion = nn.BCEWithLogitsLoss() # Needed since we are doing multilabel classification
criterion = nn.BCELoss() # Needed since we are doing multilabel classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

''' Training '''
# Train the model
print("Training model...")
total_step = len(train_loader)

best_val_acc = 0
best_val_f1 = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    scores = np.zeros(6)

    for i, (specs, labels) in enumerate(train_loader):
        # print("LABEL SHAPE:", labels.shape)
        # print("SPECT SHAPE:", specs.shape)
        # Forward pass
        outputs = model(specs.float().to(device))["clipwise_output"].to(device)
        labels = labels.to(device)
        loss = criterion(outputs, labels).float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        predicted = outputs > threshold

        epoch_total += batch_size
        epoch_correct += np.sum(
            (
                predicted.cpu().numpy() == labels.cpu().numpy()
            ).all(1)
        )

        # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {f1 * 100}%")
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {epoch_correct/epoch_total}")

        #scores = np.add(scores, f1 * batch_size)
        epoch_loss += batch_size * loss.item()

    writer.add_scalar("Loss/train", epoch_loss/epoch_total, epoch)
    writer.add_scalar("Accuracy/train", epoch_correct/epoch_total, epoch)
    
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {scores / epoch_total * 100}%")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_correct / epoch_total * 100}%")

    ''' Validation '''
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        acc_total = 0

        total = 0

        val_scores = np.zeros(6)
        val_loss = 0

        for specs, labels in validation_loader:
            outputs = model(specs.float().to(device))["clipwise_output"].to(device)
            labels = labels.to(device)
            loss = criterion(outputs, labels).float()

            batch_size = labels.shape[0]
            total += batch_size
            predicted = outputs > threshold

            correct += np.sum(
                (
                    predicted.cpu().numpy() == labels.cpu().numpy()
                ).all(1)
            )
            val_loss += batch_size * loss.item()

        print(f'Validation Accuracy: {correct / total * 100}%')
        # write to csv
        #csv.write(f"{epoch+1},{epoch_loss},{','.join(map(str, scores / epoch_total * 100))},{','.join(map(str, val_scores / total * 100))}\n")
        csv.write(f"{epoch+1},{epoch_loss},{epoch_correct / epoch_total * 100},{correct / total * 100}\n")
        csv.flush()
        os.fsync(csv.fileno())

        writer.add_scalar("Loss/val", val_loss / total, epoch)
        writer.add_scalar("Accuracy/val", correct / total, epoch)
        '''writer.add_scalar("F1/val", {
            "I": val_scores[0],
            "E": val_scores[1],
            "D": val_scores[2],
            "R": val_scores[3],
            "W": val_scores[4],
            "S": val_scores[5],
        }, epoch)'''

        # Save the model checkpoint if validation f1 improves
        if correct/total > best_val_acc:
            print('Saving model...')
            torch.save(model.state_dict(), 'model_fsd.pt')
            #best_val_acc = correct / total
            best_val_acc = correct/total
    
    writer.close()