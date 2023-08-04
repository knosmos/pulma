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
from models import TwinNetworkDeconv, Cnn14_8k
import people_split
#import librosa
#import librosa.display

import time
import sys
import argparse

''' Settings '''
# Argparser
parser = argparse.ArgumentParser(description='Train PANN + CPS on the HF Lung dataset')

# Experiment/Data
parser.add_argument('--name', type=str, default="exp", help='Name of the experiment')
parser.add_argument('--percent', type=float, default=25, help='Percentage of supervised data to use')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint to resume training from')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')

parser.add_argument('--val', type=float, default=0.1, help='Validation split')
parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
parser.add_argument('--cps', action='store_true', help='Use CPS loss')

# NN Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--batch', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')


# Get arguments
args = parser.parse_args()
batch_size = args.batch
validation_split = args.val
shuffle_dataset = args.shuffle

print(["Not using CPS", "Using CPS"][args.cps])
print(f"Using {args.percent}% of labeled data")
print(["Not using pretrained model", f"Using pretrained model"][args.pretrained])

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)
np.random.seed(random_seed)

threshold = 0.4

''' Data '''
print("Loading data...")

# Load data
train_dataset = HF_Lung_Dataset(train=True, dataFName='train_steth_combined.h5')

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
split = int(np.floor(validation_split * dataset_size))
people = people_split.people
if shuffle_dataset:    
    np.random.shuffle(people)
print("number of people:", len(people))
indices = []
for person in people:
    indices.extend(person)

train_indices, val_indices = indices[split:], indices[:split]
#train_indices, val_indices = indices[:dataset_size-split], indices[dataset_size-split:]

# extract percentage of data
train_labeled_indices = train_indices[:int(len(train_indices) * args.percent) // 100]
train_unlabeled_indices = train_indices[int(len(train_indices) * args.percent) // 100:]

# Creating PT data samplers and loaders:
train_labeled_sampler = SubsetRandomSampler(train_labeled_indices)
train_unlabeled_sampler = SubsetRandomSampler(train_unlabeled_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_labeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_labeled_sampler)
train_unlabeled_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_unlabeled_sampler)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

print(f"Train: {len(train_indices)} samples, Validation: {len(val_indices)} samples")

# Result storage
csv = open(f"results/{args.name}_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")
writer = SummaryWriter(comment="_"+args.name)

''' Model '''
# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize models
print("Initializing model...")
#extractor = Cnn14_8k(sample_rate=8000, window_size=256, hop_size=80, mel_bins=64, fmin=50, fmax=4000, classes_num=527)
model = TwinNetworkDeconv(64, 4, device, single=(not args.cps))

# Load pretrained model
if args.pretrained:
    print("Loading pretrained model")
    checkpoint_path = "Cnn14_8k_mAP=0.416.pth"
    checkpoint = torch.load(checkpoint_path)
    
    # Eliminate mismatching layers
    model_dict = model.l.state_dict()
    state_dict={k : v for k, v in zip(model_dict.keys(), checkpoint['model'].values()) if v.size() == model_dict[k].size()}
    
    print(model_dict.keys())
    print(state_dict.keys())

    model.l.load_state_dict(state_dict, strict=False)
    model.r.load_state_dict(state_dict, strict=False)

# Load from checkpoint
if args.checkpoint != "":
    print("Loading checkpoint")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    #optimizer.load_state_dict(checkpoint['optimizer'])

model.to(device)
#model = torch.nn.DataParallel(model)

# Loss and optimizer
criterion = nn.BCELoss() # Needed since we are doing multilabel classification
criterion_cps = nn.BCELoss()

# these are copied from CPS code
#criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
#                                    min_kept=pixel_num, use_weight=False)
#criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

optim_l = optim.AdamW(model.l.parameters(), lr=args.lr, weight_decay=0.01)
optim_r = optim.AdamW(model.r.parameters(), lr=args.lr, weight_decay=0.01)

lr_scheduler_l = optim.lr_scheduler.PolynomialLR(optim_l, power=0.9, total_iters=args.epochs)
lr_scheduler_r = optim.lr_scheduler.PolynomialLR(optim_r, power=0.9, total_iters=args.epochs)

''' Training '''
# Train the model
print("Training model...")

total_step = max(len(train_unlabeled_loader), len(train_labeled_loader))

best_val_acc = 0
best_val_loss = 100
best_val_f1 = 0

for epoch in range(args.epochs):
    model.train()

    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    epoch_correct_u = 0 # unlabeled
    epoch_total_u = 0

    sum_sup_loss = 0
    sum_unsup_thresholded_loss = 0
    sum_unsup_loss = 0
    sum_cps_loss = 0

    scores = np.zeros(4)
    normal_total = 0
    #for i, (specs_labeled, labels, specs_unlabeled, _) in enumerate(zip(train_labeled_loader, train_unlabeled_loader)):
    
    for i in range(total_step):
        # Get data
        specs_labeled, labels = train_labeled_loader.__iter__().__next__()
        specs_unlabeled, unsup_labels = train_unlabeled_loader.__iter__().__next__()

        #print("LABEL SHAPE:", labels.shape)
        #print("SPECT SHAPE:", specs.shape)

        # Forward pass: supervised
        inp = specs_labeled.float().to(device)
        #embeddings = extractor(inp).to(device)
        #pred_sup_l, pred_sup_r = model(embeddings)
        pred_sup_l, pred_sup_r = model(inp)
        sup_loss =  criterion(
                        pred_sup_l.to(device),
                        labels.to(device)
                    ).float() + \
                    criterion(
                        pred_sup_r.to(device),
                        labels.to(device)
                    ).float()

        #writer.add_scalar("Loss/sup", sup_loss, epoch)
        sum_sup_loss += sup_loss

        # Track F1 score
        predicted = pred_sup_l > threshold
        batch_size = labels.shape[0]
        #epoch_total += batch_size

        correct = 0
        total = labels.shape[0] * labels.shape[1]
        epoch_total += total

        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            correct += same.all(axis=1).sum()
        epoch_correct += correct

        # Forward pass: unsupervised
        inp = specs_unlabeled.float().to(device)
        #embeddings = extractor(inp).to(device)
        #pred_unsup_l, pred_unsup_r = model(embeddings)
        pred_unsup_l, pred_unsup_r = model(inp)

        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)

        '''
        _, max_l = torch.max(pred_l, dim=2)
        _, max_r = torch.max(pred_r, dim=2)
        #print(max_l)
        max_l = max_l.long()
        max_r = max_r.long()
        '''

        max_l = pred_l > threshold
        max_r = pred_r > threshold
        max_l = max_l.float()
        max_r = max_r.float()

        #print("dim of target:", max_l.shape, max_r.shape)
        #print("dim of pred:", pred_l.shape, pred_r.shape)

        if args.cps:
            cps_loss = criterion_cps(pred_l, max_r).float() + criterion_cps(pred_r, max_l).float()
            #cps_loss = criterion_cps(pred_l.transpose(1,2), max_r) + criterion_cps(pred_r.transpose(1,2), max_l)
            #writer.add_scalar("Loss/cps", cps_loss, epoch)
            sum_cps_loss += cps_loss

        # calculate accuracy on unlabeled data
        predicted = pred_unsup_l > threshold
        batch_size = unsup_labels.shape[0]
        #epoch_total += batch_size

        correct = 0
        total = unsup_labels.shape[0] * unsup_labels.shape[1]
        epoch_total_u += total

        for image in range(len(predicted)):
            same = predicted[image] == unsup_labels[image].to(device)
            correct += same.all(axis=1).sum()
        epoch_correct_u += correct

        # calculate (but don't use) unlabeled loss
        unlabeled_sup_loss = criterion(pred_unsup_l, unsup_labels.to(device)).float() + \
                                criterion(pred_unsup_r, unsup_labels.to(device)).float()
                                
        unlabeled_sup_thresholded_loss = criterion((pred_unsup_l > threshold).float(), unsup_labels.to(device)).float() + \
                                criterion((pred_unsup_r > threshold).float(), unsup_labels.to(device)).float()

        #writer.add_scalar("Loss/sup_unlabeled", unlabeled_sup_loss, epoch)
        #writer.add_scalar("Loss/sup_unlabeled_thresholded", unlabeled_sup_thresholded_loss, epoch)
        sum_unsup_loss += unlabeled_sup_loss
        sum_unsup_thresholded_loss += unlabeled_sup_thresholded_loss

        #loss = sup_loss + 1.5 * cps_loss
        loss = sup_loss
        if args.cps:
            loss += 1.0 * cps_loss

        # Backward and optimize
        optim_l.zero_grad()
        optim_r.zero_grad()
        loss.backward()
        optim_l.step()
        optim_r.step()

        if (i+1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct/total * 100}%")
        # print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {f1 * 100}%")
        #print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct/total * 100}%")

        #scores = np.add(scores, f1 * batch_size)
        epoch_loss += loss.item() * batch_size
        normal_total += batch_size
    
    # Modify learning rate
    lr_scheduler_l.step()
    lr_scheduler_r.step()

    #print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {scores / epoch_total * 100}%")
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / normal_total:.4f}, Accuracy: {epoch_correct / epoch_total * 100}%")

    writer.add_scalar("Loss/train", epoch_loss / normal_total, epoch)
    writer.add_scalar("Accuracy/train_labeled", epoch_correct / epoch_total, epoch)
    writer.add_scalar("Accuracy/train_unlabeled", epoch_correct_u / epoch_total_u, epoch)

    if args.cps:
        writer.add_scalar("Loss/sup", sum_sup_loss / total_step, epoch)
        writer.add_scalar("Loss/sup_unlabeled", sum_unsup_loss / total_step, epoch)
        writer.add_scalar("Loss/sup_unlabeled_thresholded", sum_unsup_thresholded_loss / total_step, epoch)
        writer.add_scalar("Loss/cps", sum_cps_loss / total_step, epoch)

    ''' Validation '''
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        acc_total = 0

        total = 0

        val_scores = np.zeros(4)
        val_loss = 0

        full_outputs = "empty"
        full_labels = "empty"

        for specs, labels in validation_loader:
            inp = specs.float().to(device)
            #embeddings = extractor(inp).to(device)
            #outputs, _ = model(embeddings)
            outputs, _ = model(inp)
            loss = criterion(outputs, labels.to(device)).float()

            if full_outputs == "empty":
                full_outputs = outputs
            else: # concat
                full_outputs = torch.cat((full_outputs, outputs))

            if full_labels == "empty":
                full_labels = labels
            else:
                full_labels = torch.cat((full_labels, labels))

            batch_size = labels.shape[0]
            total += batch_size
            predicted = outputs > threshold

            for image in range(len(predicted)):
                same = predicted[image] == labels[image].to(device)
                correct += same.all(axis=1).sum()
            acc_total += labels.shape[0] * labels.shape[1]

            predicted = predicted.reshape(
                (labels.shape[0] * labels.shape[1], 4)
            ).cpu().numpy().transpose(1, 0)
            
            labels = labels.reshape(
                (labels.shape[0] * labels.shape[1], 4)
            ).cpu().numpy().transpose(1, 0)

            for i in range(4):
                val_scores[i] += batch_size * sklearn.metrics.f1_score(
                    predicted[i],
                    labels[i],
                    zero_division=0
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
        score_sum = np.sum(val_scores) / total
        writer.add_scalar("Accuracy/val_f1_sum", score_sum, epoch)
        if score_sum > best_val_f1:
            print('Saving model...')
            torch.save(model.state_dict(), f"model_{args.name}.pt")
            #best_val_acc = correct / total
            best_val_f1 = score_sum

        # Save model checkpoint if val loss is lower
        if val_loss < best_val_loss:
            print('Saving model...')
            torch.save(model.state_dict(), f"model_{args.name}_loss.pt")
            best_val_loss = val_loss
