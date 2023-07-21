print("Loading libraries...")

''' Imports '''
# Pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# Calculate performance
import sklearn

# Visualization imports
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom imports
from H5_dataset import HF_Lung_Dataset
from models import Cnn14_8k, TwinNetwork

import sys

''' Settings '''
# NN Hyperparameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
validation_split = .1
shuffle_dataset = False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)

threshold = 0.4

# Tensorboard
writer = SummaryWriter()

''' Data '''
print("Loading data...")

# Load data
test_dataset = HF_Lung_Dataset(train=False, dataFName='test_steth_combined.h5')

# Creating data indices for training and validation splits:
dataset_size = len(test_dataset)
indices = list(range(dataset_size))

test_sampler = SubsetRandomSampler(indices)

test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
print(f"Test: {len(indices)} samples")

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

''' Model '''
# Initialize model
print("Initializing model...")
model = TwinNetwork(64, 4, device)
#fc_audioset_in_params = model.fc_audioset.in_features
#model.fc_audioset = nn.Linear(fc_audioset_in_params, 6, bias=True)

exp_name = sys.argv[1]

checkpoint = torch.load(f"model_{exp_name}.pt")

# Remove all the "module." from the keys because that's there for some reason
checkpoint_clean = {}
for i in checkpoint.keys():
    checkpoint_clean[i.replace("module.", "")] = checkpoint[i]

model.load_state_dict(checkpoint_clean)

model.to(device)

''' Testing '''
# Train the model
print("Testing model...")
with torch.no_grad():
    scores = np.zeros(4)
    bscores = np.zeros(4)
    total = 0
    correct = 0
    acc_total = 0

    full_outputs = "empty"
    full_labels = "empty"

    for specs, labels in tqdm(test_loader):
        inp = specs.float().to(device)
        outputs, _ = model(inp)
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
        
        #print(predicted.shape)

        #print(predicted, labels)
        #same = predicted == labels.to(device)
        #same = same.sum(axis=1).sum(axis=0).cpu().numpy()
        #correct = np.add(correct, same)

        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            correct += same.all(axis=1).sum()
        acc_total += labels.shape[0] * labels.shape[1]

        # calculate F1 score for each of six classes
        predicted = predicted.reshape(
            (labels.shape[0] * labels.shape[1], 4)
        ).transpose(1, 0)
        labels = labels.reshape(
            (labels.shape[0] * labels.shape[1], 4)
        ).transpose(1, 0)
        #print(labels.shape)
        for i in range(4):
            scores[i] += batch_size * sklearn.metrics.f1_score(
                predicted[i].cpu().numpy(),
                labels[i].cpu().numpy(),
                zero_division=0
            )
            bscores[i] += batch_size * sklearn.metrics.fbeta_score(
                predicted[i].cpu().numpy(),
                labels[i].cpu().numpy(),
                beta=2,
                zero_division=0
            )

    #writer.add_pr_curve(exp_name + '_pr_curve', full_labels, full_outputs, 0)
    #writer.close()

    print(f'Test F1 Score: {(scores / total) * 100}%')
    print(f'Test F2 Score: {(bscores / total) * 100}%')
    print(f'Test Accuracy: {correct / acc_total * 100}%')

    print(full_labels.shape, full_outputs.shape) 
    full_labels = full_labels.reshape(
        (full_labels.shape[0] * full_labels.shape[1], 4)
    ).transpose(1, 0).cpu().numpy()
    full_outputs = full_outputs.reshape(
        (full_outputs.shape[0] * full_outputs.shape[1], 4)
    ).transpose(1, 0).cpu().numpy()

    #fig = plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")

    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")

    fig.suptitle(exp_name)
    plt.style.use("ggplot")

    auc = []
    f1 = []

    for i in range(4):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(full_labels[i], full_outputs[i])
        ax1.plot(recall, precision, label="IEDC"[i])
        f1_scores = 2*recall*precision/(recall+precision+np.finfo(float).eps)
        print(f'Best threshold for class {i}: ', thresholds[np.argmax(f1_scores)])
        print(f'Best F1-Score for class {i}: ', np.max(f1_scores))

        #p = full_labels[i].sum()
        #n = len(full_labels[i]) - p
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(full_labels[i], full_outputs[i])
        '''
        tp = tpr * p
        fp = fpr * n
        prec = tp / (fp + tp)
        rec = tpr
        '''
        ax2.plot(fpr, tpr, label="IEDC"[i])
        auc.append(sklearn.metrics.roc_auc_score(full_labels[i], full_outputs[i]))
        f1.append(sklearn.metrics.f1_score(full_labels[i] > threshold, full_outputs[i].round()))
    
    print(f'AUC: {auc}')
    print(f'F1: {f1}')
    
    plt.legend()
    plt.savefig(f"pr_{exp_name}_verify.png")