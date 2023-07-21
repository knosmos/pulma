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
from models import Cnn14_DecisionLevelMax
import librosa
import librosa.display
from dataset_gen import load_sample

''' Load model '''
print("Loading model state...")
model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)

fc1_in_params = model.fc1.in_features
fc1_out_params = model.fc1.out_features

fc_audioset_in_params = model.fc_audioset.in_features

# Rewrite last layers
model.fc1 = nn.Linear(fc1_in_params, fc1_out_params, bias=True)
model.fc_audioset = nn.Linear(fc_audioset_in_params, 6, bias=True) # 6 output classes

checkpoint = torch.load("model_freeze.pt")

# Remove all the "module." from the keys because that's there for some reason
checkpoint_clean = {}
for i in checkpoint.keys():
    checkpoint_clean[i.replace("module.", "")] = checkpoint[i]

model.load_state_dict(checkpoint_clean)

''' Settings '''
folder = "../data/test/"
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
ctr = 0

for file in tqdm(os.listdir(folder)):
    if not file.endswith(".wav"):
        continue

    ctr = (ctr + 1) % 10    
    if ctr % 10 != 1:
        continue

    if not file in ["steth_20190815_14_18_44.wav", "steth_20191025_00_39_26.wav", "trunc_2019-08-21-10-05-00-L1_14.wav"]:
        continue

    # Set up plot
    fig, (ax_raw, ax_spect, ax_annot, ax_ground) = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    fig.suptitle("PANN - Pretrained, Finetuned: " + file)

    # Remove x axis labels
    ax_raw.xaxis.set_visible(False)
    ax_spect.xaxis.set_visible(False)
    ax_annot.xaxis.set_visible(False)
    ax_ground.xaxis.set_visible(False)

    # Remove y axis labels
    ax_raw.yaxis.set_visible(False)
    ax_spect.yaxis.set_visible(False)
    ax_annot.yaxis.set_visible(False)
    ax_ground.yaxis.set_visible(False)

    # Get raw audio
    data, sr = librosa.load(folder+file)
    t = np.linspace(0, len(data) / sr, num=len(data))
    ax_raw.plot(t, data, color="grey", alpha=0.5, zorder=0, label="Audio")

    # Get spectrogram
    y, spect = load_sample(folder+file)

    img = librosa.display.specshow(librosa.power_to_db(spect.transpose()),
        x_axis='time',
        y_axis='mel', sr=32000, hop_length=320,
        ax=ax_spect,
        cmap="viridis",
    )

    # run model
    data = torch.from_numpy(spect[None,...]).float()
    outputs = model(data)
    outputs = outputs["framewise_output"].detach().numpy()[0]

    # get events
    events = []
    for i in range(6):
        st = -1
        for j in range(len(outputs)):
            if outputs[j, i] > 0.4:
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
        start = start * 320 / 32000
        end = end * 320 / 32000
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
        for j in range(len(y)):
            if y[j, i] == 1:
                if st == -1:
                    st = j
            else:
                if st != -1:
                    events.append([mapping[i], st, j])
                    st = -1
        if st != -1:
            events.append([mapping[i], st, len(y)])

    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 320 / 32000
        end = (end-1) * 320 / 32000
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

    plt.savefig("disp/freeze_"+file[:-4]+".png")
    #plt.show()
    plt.close()