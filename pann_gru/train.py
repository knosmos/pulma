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
from models import Cnn14_8k, BiGRUNetwork
import librosa
import librosa.display

import time
import sys
import argparse

''' Settings '''
# Argparser
parser = argparse.ArgumentParser(description='Train PANN/GRU on the HF Lung dataset')

# Experiment/Data
parser.add_argument('--name', type=str, default="exp", help='Name of the experiment')
parser.add_argument('--percent', type=int, default=100, help='Percentage of data to use')
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
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_steth.h5')

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# extract percentage of data
train_indices = train_indices[:(len(train_indices) * percent_data) // 100]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

# Result storage
csv = open(f"results/{experiment_name}_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")
writer = SummaryWriter()

''' Model '''
# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize models
print("Initializing model...")
extractor = Cnn14_8k(sample_rate=8000, window_size=256, hop_size=80, mel_bins=64, fmin=50, fmax=4000, classes_num=6)
model = BiGRUNetwork(2048, 6, device)

# Load pretrained model
print("Loading pretrained model")
#checkpoint_path = "Cnn14_8k_mAP=0.416.pth"
checkpoint_path = "model_finetune.pt"
checkpoint = torch.load(checkpoint_path)
#extractor.load_state_dict(checkpoint['model'])
extractor.load_state_dict(checkpoint)

# Freeze pretrained parameters
for param in extractor.parameters():
    param.requires_grad = False

extractor.to(device)

# Load from checkpoint
if model_checkpoint != "":
    print("Loading checkpoint")
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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
best_val_loss = 100
best_val_f1 = 0

for epoch in range(num_epochs):
    model.train()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    scores = np.zeros(6)
    normal_total = 0
    for i, (specs, labels) in enumerate(train_loader):
        # print("LABEL SHAPE:", labels.shape)
        # print("SPECT SHAPE:", specs.shape)
        # Forward pass
        inp = specs.float().to(device)
        embeddings = extractor(inp).to(device)
        outputs = model(embeddings).to(device)
        loss = criterion(outputs, labels.to(device)).float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track F1 score
        predicted = outputs > threshold

        batch_size = labels.shape[0]
        #epoch_total += batch_size

        correct = 0
        total = labels.shape[0] * labels.shape[1]
        epoch_total += total
        normal_total += batch_size

        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            correct += same.all(axis=1).sum()
        epoch_correct += correct

        # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {f1 * 100}%")
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct/total * 100}%")

        #scores = np.add(scores, f1 * batch_size)
        epoch_loss += loss.item() * batch_size
    
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {scores / epoch_total * 100}%")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / normal_total:.4f}, Accuracy: {epoch_correct / epoch_total * 100}%")

    writer.add_scalar("Loss/train", epoch_loss / normal_total, epoch)
    writer.add_scalar("Accuracy/train", epoch_correct / epoch_total, epoch)

    ''' Validation '''
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        acc_total = 0

        total = 0

        val_scores = np.zeros(6)
        val_loss = 0

        for specs, labels in validation_loader:
            inp = specs.float().to(device)
            embeddings = extractor(inp).to(device)
            outputs = model(embeddings).to(device)
            loss = criterion(outputs, labels.to(device)).float()

            batch_size = labels.shape[0]
            total += batch_size
            predicted = outputs > threshold

            for image in range(len(predicted)):
                same = predicted[image] == labels[image].to(device)
                correct += same.all(axis=1).sum()
            acc_total += labels.shape[0] * labels.shape[1]

            predicted = predicted.reshape(
                (labels.shape[0] * labels.shape[1], 6)
            ).cpu().numpy().transpose(1, 0)
            
            labels = labels.reshape(
                (labels.shape[0] * labels.shape[1], 6)
            ).cpu().numpy().transpose(1, 0)

            for i in range(6):
                val_scores[i] += batch_size * sklearn.metrics.f1_score(
                    predicted[i],
                    labels[i]
                )
            
            val_loss += loss.item() * batch_size

        val_loss /= total
        print(f'Validation F1: {val_scores / total * 100}%')
        print(f'Validation Accuracy: {correct / acc_total * 100}%')

        writer.add_scalar("Accuracy/val", correct / acc_total, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # write to csv
        #csv.write(f"{epoch+1},{epoch_loss},{','.join(map(str, scores / epoch_total * 100))},{','.join(map(str, val_scores / total * 100))}\n")
        csv.write(f"{epoch+1},{epoch_loss},{epoch_correct / epoch_total * 100},{correct / acc_total * 100},{','.join(map(str, val_scores / total * 100))}\n")
        csv.flush()
        os.fsync(csv.fileno())

        # Save the model checkpoint if validation accuracy improves
        if np.sum(val_scores) > best_val_f1:
            print('Saving model...')
            torch.save(model.state_dict(), f"model_{experiment_name}.pt")
            #best_val_acc = correct / total
            best_val_f1 = np.sum(val_scores)

        # Save model checkpoint if val loss is lower
        if val_loss < best_val_loss:
            print('Saving model...')
            torch.save(model.state_dict(), f"model_{experiment_name}_loss{val_loss:.2f}.pt")
            best_val_loss = val_loss
