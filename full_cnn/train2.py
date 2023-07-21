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
import sklearn

# Custom imports
#from custom_dataset import HF_Lung_Dataset
from H5_dataset import HF_Lung_Dataset
from model_simple import Baseline
import librosa
import librosa.display

import time

''' Settings '''
# NN Hyperparameters
num_epochs = 300
batch_size = 256 #64
learning_rate = 0.001 #0.01
validation_split = .3
shuffle_dataset = True #False # Prevent train/val mixing

# Other settings
random_seed = 4 # chosen by fair dice roll. guaranteed to be random.
torch.manual_seed(random_seed)

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
model = Baseline(length=938, n_mels=128, num_classes=6)

# Enable GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Loss and optimizer
#criterion = nn.BCEWithLogitsLoss() # Needed since we are doing multilabel classification
criterion = nn.BCELoss() # Needed since we are doing multilabel classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

''' Display '''
colors = {
    "I": ["crimson", 0], # Inhalation
    "E": ["cornflowerblue", 1], # Exhalation
    "D": ["mediumseagreen", 2], # Discontinuous adventitious sound?
    "Rhonchi": ["slategrey",3],
    "Wheeze": ["indigo",4],
    "Stridor": ["orange",5]
}
mapping = ["I", "E", "D", "Rhonchi", "Wheeze", "Stridor"]
plt.style.use("ggplot")

def display(spect, outputs, labels):
    # Set up figure
    fig, (ax_spect, ax_annot, ax_ground) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    # Remove x axis labels
    ax_spect.xaxis.set_visible(False)
    ax_annot.xaxis.set_visible(False)
    ax_ground.xaxis.set_visible(False)

    # Remove y axis labels
    ax_spect.yaxis.set_visible(False)
    ax_annot.yaxis.set_visible(False)
    ax_ground.yaxis.set_visible(False)

    # Spectrogram
    img = librosa.display.specshow(librosa.power_to_db(np.transpose(spect.cpu().numpy()), ref=np.max),
        x_axis='time',
        y_axis='mel', sr=4000,
        ax=ax_spect,
        cmap="viridis",
    )

    # get events
    events = []
    for i in range(6):
        st = -1
        for j in range(len(outputs)):
            if outputs[j, i] == 1:
                if st == -1:
                    st = j
            else:
                if st != -1:
                    events.append([mapping[i], st, j])
                    st = -1
        if st != -1:
            events.append([mapping[i], st, len(outputs)])

    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 64 / 4000
        end = end * 64 / 4000
        ax_annot.axvspan(
            start,
            end,
            vert_size*colors[label][1],
            vert_size*(colors[label][1]+1),
            alpha=0.6,
            color=colors[label][0],
            zorder = 10,
            label=label
        )

    # Get ground truth
    events = []
    for i in range(6):
        st = -1
        for j in range(len(labels)):
            if labels[j, i] == 1:
                if st == -1:
                    st = j
            else:
                if st != -1:
                    events.append([mapping[i], st, j])
                    st = -1
        if st != -1:
            events.append([mapping[i], st, len(labels)])
    #print(events)
    #if events == []:
       # print(labels)
    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 64 / 4000
        end = (end-1) * 64 / 4000
        ax_ground.axvspan(
            start,
            end,
            vert_size*colors[label][1],
            vert_size*(colors[label][1]+1),
            alpha=0.6,
            color=colors[label][0],
            zorder = 10,
            label=label
        )

    plt.xlim([0, 15])

    handles, labels = ax_annot.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_annot.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)

    handles, labels = ax_ground.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_ground.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

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
        #print("Label Size!!!!", labels.shape)
        # Forward pass
        outputs = model(specs.float().to(device)).to(device)
        loss = criterion(outputs, labels.to(device)).float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.shape[0] * labels.shape[1]
        predicted = outputs > threshold
        #correct = (predicted == labels.to(device)).all(axis=1).sum().item()
        correct = 0
        for image in range(len(predicted)):
            same = predicted[image] == labels[image].to(device)
            my_correct = same.all(axis=1).sum()
            correct += my_correct
            #print("train image accuracy: ", my_correct / predicted.shape[1] * 100)
            #print(f'labels = {sum(labels[image])}')
            #if image == 0:
            #    display(specs[image], predicted[image], labels[image])
        #print(f'predicted={predicted}, labels={labels}, correct = {correct}')

        #if (i+1) % 10 == 0:
        if (i+1) % 1 == 0:
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
        acc_total = 0

        total = 0

        val_scores = np.zeros(6)

        for specs, labels in validation_loader:
            outputs = model(specs.float().to(device)).to(device)

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

        print(f'Validation F1: {val_scores / total * 100}%')
        print(f'Validation accuracy: {(correct / acc_total) * 100:.2f}%')
        
        # write to csv
        csv.write(f"{epoch+1},{epoch_loss},{epoch_correct / epoch_total * 100},{correct / acc_total * 100},{','.join(map(str, val_scores / total * 100))}\n")
        csv.flush()
        os.fsync(csv.fileno())

        # Save the model checkpoint if validation accuracy improves
        if (correct / total) > best_val_acc:
            print('Saving model...')
            torch.save(model.state_dict(), 'model.pt')
            best_val_acc = correct / total
