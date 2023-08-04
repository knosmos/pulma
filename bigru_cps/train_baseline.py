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
from models_baseline import TwinNetworkDeconv
import people_split
#import librosa
#import librosa.display

import time
import sys
import argparse

''' Settings '''
# Argparser
parser = argparse.ArgumentParser(description='Train PANN baseline on the HF Lung dataset')

# Experiment/Data
parser.add_argument('--name', type=str, default="exp", help='Name of the experiment')
parser.add_argument('--percent', type=float, default=25, help='Percentage of supervised data to use')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint to resume training from')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')

parser.add_argument('--val', type=float, default=0.1, help='Validation split')
parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
parser.add_argument('--cps', action='store_true', help='Use CPS loss')
parser.add_argument('--cps_weight', type=float, default=1.0, help='Weight of CPS loss')
parser.add_argument('--warmup', type=int, default=0, help='Number of epochs to run purely supervised')

parser.add_argument('--seed', type=int, default=4, help='Random seed')

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
print(f"Using {args.warmup} warmup epochs")

# Other settings
random_seed = args.seed # chosen by fair dice roll. guaranteed to be random.
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

''' Model '''
# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize models
print("Initializing model...")
#extractor = Cnn14_8k(sample_rate=8000, window_size=256, hop_size=80, mel_bins=64, fmin=50, fmax=4000, classes_num=527)
torch.manual_seed(42) # NOTE DIFFERENT MODEL INITIALIZATION
model = TwinNetworkDeconv(64, 4, device, single=(not args.cps))

# Load pretrained model
if args.pretrained:
    print("Loading pretrained model")
    checkpoint_path = "Cnn14_8k_mAP=0.416.pth"
    checkpoint = torch.load(checkpoint_path)
    
    # Eliminate mismatching layers
    model_dict = model.l.backbone.state_dict()
    state_dict={k : v for k, v in zip(model_dict.keys(), checkpoint['model'].values()) if v.size() == model_dict[k].size()}

    model.l.backbone.load_state_dict(state_dict, strict=False)
    model.r.backbone.load_state_dict(state_dict, strict=False)

# Load from checkpoint
if args.checkpoint != "":
    print("Loading checkpoint")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    #optimizer.load_state_dict(checkpoint['optimizer'])

model.to(device)
#model = torch.nn.DataParallel(model)

wt = np.zeros(4)
for specs, label_batch in train_labeled_loader:
    wt = np.add(wt, label_batch.sum(dim=1).sum(dim=0))
m = int(wt[0])
for i in range(4):
    wt[i] = m / wt[i]
print("Loss weighting:", wt)

# Loss and optimizer
#criterion = nn.BCELoss(weight=wt.to(device)) # Needed since we are doing multilabel classification
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.2882, 5.0703, 7.5007, 7.1268]).to(device))
#criterion_cps = nn.BCELoss(reduction="none")
criterion_cps = nn.BCELoss(weight=wt.to(device))

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

# Result storage
csv = open(f"results/{args.name}_{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.csv", "w")
writer = SummaryWriter(comment = "_"  + args.name)

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

    for i in tqdm(range(total_step)):
        #print(f'iter={i}')
        optim_l.zero_grad()
        optim_r.zero_grad()
        # Get data
        specs_labeled, labels = train_labeled_loader.__iter__().__next__()
        specs_unlabeled, unsup_labels = train_unlabeled_loader.__iter__().__next__()
        #specs_labeled, labels = train_labeled_loader.next()
        #specs_unlabeled, unsup_labels = train_unlabeled_loader.next()
        
        #print("LABEL SHAPE:", labels.shape)
        #print("SPECT SHAPE:", specs.shape)

        # Forward pass: supervised
        inp = specs_labeled.float().to(device)
        #embeddings = extractor(inp).to(device)
        #pred_sup_l, pred_sup_r = model(embeddings)
        (pred_sup_l_f, pred_sup_l), (pred_sup_r_f, pred_sup_r) = model(inp)
        sup_loss =  criterion(
                        pred_sup_l.to(device),
                        labels.to(device)
                    ).float() + \
                    criterion(
                        pred_sup_r.to(device),
                        labels.to(device)
                    ).float()
        loss = sup_loss
        sum_sup_loss += sup_loss * batch_size

        # Track accuracy
        predicted = pred_sup_l_f > threshold
        batch_size = labels.shape[0]

        correct = 0
        total = labels.shape[0] * labels.shape[1]
        epoch_total += total

        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            correct += same.all(axis=1).sum()
        epoch_correct += correct

        if args.cps and epoch >= args.warmup:
            # Forward pass: unsupervised
            inp = specs_unlabeled.float().to(device)
            #embeddings = extractor(inp).to(device)
            #pred_unsup_l, pred_unsup_r = model(embeddings)
            pred_unsup_l, pred_unsup_r = model(inp)

            pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)

            max_l = pred_l > threshold
            max_r = pred_r > threshold
            max_l = max_l.float()
            max_r = max_r.float()
            
            cps_loss = criterion_cps(pred_l, max_r).float() + criterion_cps(pred_r, max_l).float()
            

            # calculate accuracy on unlabeled data
            predicted = pred_unsup_l > threshold

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

            sum_unsup_loss += unlabeled_sup_loss * batch_size
            sum_unsup_thresholded_loss += unlabeled_sup_thresholded_loss * batch_size

            #print(cps_loss)
            loss += args.cps_weight * cps_loss
            sum_cps_loss += cps_loss * batch_size

        # Backward and optimize
        loss.backward()
        optim_l.step()
        optim_r.step()

        #if (i+1) % 30 == 0:
        #    print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct/total * 100}%")
        
        epoch_loss += loss.item() * batch_size
        normal_total += batch_size
    
    # Modify learning rate
    lr_scheduler_l.step()
    lr_scheduler_r.step()

    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss / normal_total:.4f}, Accuracy: {epoch_correct / epoch_total * 100}%")

    writer.add_scalar("Loss/train", epoch_loss / normal_total, epoch)
    writer.add_scalar("Loss/sup", sum_sup_loss / normal_total, epoch)
    writer.add_scalar("Accuracy/train_labeled", epoch_correct / epoch_total, epoch)
    if args.cps and epoch >= args.warmup:
        writer.add_scalar("Accuracy/train_unlabeled", epoch_correct_u / epoch_total_u, epoch)
        writer.add_scalar("Loss/cps", sum_cps_loss / normal_total, epoch)
        writer.add_scalar("Loss/sup_unlabeled", sum_unsup_loss / normal_total, epoch)
        writer.add_scalar("Loss/sup_unlabeled_thresholded", sum_unsup_thresholded_loss / normal_total, epoch)

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
            (outputs, outputs_nosigmoid), (_, _) = model(inp)
            loss = criterion(outputs_nosigmoid, labels.to(device)).float()

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

            val_loss += loss.item() * batch_size

        full_labels = full_labels.reshape(
            (full_labels.shape[0] * full_labels.shape[1], 4)
        ).transpose(1, 0).cpu().numpy()
        full_outputs = full_outputs.reshape(
            (full_outputs.shape[0] * full_outputs.shape[1], 4)
        ).transpose(1, 0).cpu().numpy()

        for i in range(4):
            val_scores[i] = sklearn.metrics.f1_score(
                full_labels[i],
                full_outputs[i] > threshold,
                zero_division=0
            )

        val_loss /= total
        print(f'Validation F1: {val_scores * 100}%')
        print(f'Validation Accuracy: {correct / acc_total * 100}%')

        writer.add_scalar("Accuracy/val", correct / acc_total, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # write to csv
        #csv.write(f"{epoch+1},{epoch_loss},{','.join(map(str, scores / epoch_total * 100))},{','.join(map(str, val_scores / total * 100))}\n")
        csv.write(f"{epoch+1},{epoch_loss},{epoch_correct / epoch_total * 100},{correct / acc_total * 100},{','.join(map(str, val_scores / total * 100))}\n")
        csv.flush()
        os.fsync(csv.fileno())

        # Save the model checkpoint if validation accuracy improves
        score_sum = np.sum(val_scores)
        writer.add_scalar("Accuracy/val_f1_sum", score_sum, epoch)
        if score_sum > best_val_f1:
            print('Saving model (f1) ...')
            torch.save(model.state_dict(), f"model_{args.name}.pt")
            #best_val_acc = correct / total
            best_val_f1 = score_sum

        '''
        # Save model checkpoint if val loss is lower
        if val_loss < best_val_loss:
            print('Saving model...')
            torch.save(model.state_dict(), f"model_{args.name}_loss.pt")
            best_val_loss = val_loss
        '''
        if correct / acc_total > best_val_acc:
            print('Saving model (acc) ...')
            torch.save(model.state_dict(), f"model_{args.name}_acc.pt")
            best_val_acc = correct / acc_total
        '''
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_{args.name}_last.pt")
        '''