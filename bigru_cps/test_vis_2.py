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
from models_deconv import TwinNetworkGRU as TwinNetwork
import librosa
import librosa.display
from dataset_gen import load_sample_combined as load_sample

''' Settings '''
folder = "../data/test/"
colors = {
    "I": ["crimson", 0], # Inhalation
    "E": ["cornflowerblue", 1], # Exhalation
    "D": ["mediumseagreen", 2], # Discontinuous adventitious sound
    "C": ["orange", 3] # Continuous adventitious sound
}
mapping = ["I", "E", "D", "C"]
plt.style.use("ggplot")

#file = "steth_20190809_15_25_43.wav"
#file = "steth_20190815_14_18_44.wav"
#file = "steth_20190815_11_34_17.wav"
'''
steth_20190809_12_02_06.wav                                                                                                                                    
steth_20190902_13_19_31.wav                                                                                                                                    
steth_20190815_09_57_12.wav                                                                                                                                    
steth_20190821_10_11_47.wav                                                                                                                                    
steth_20190815_11_34_17.wav                                                                                                                                    
steth_20190821_10_14_10.wav                                                                                                                                    
steth_20190809_11_02_39.wav                                                                                                                                    
steth_20190809_12_03_44.wav                                                                                                                                    
steth_20190810_12_38_05.wav                                                                                                                                    
steth_20190815_09_54_14.wav                                                                                                                                    
steth_20191001_00_32_24.wav                                                                                                                                    
steth_20190809_11_03_38.wav                                                                                                                                    
steth_20191027_12_03_38.wav                                                                                                                                    
steth_20191025_02_26_57.wav                                                                                                                                    
steth_20191001_00_33_09.wav                                                                                                                                    
steth_20190815_09_57_45.wav                                                                                                                                    
steth_20190902_13_20_20.wav                                                                                                                                    
steth_20190821_10_11_22.wav                                                                                                                                    
steth_20191017_13_49_30.wav                                                                                                                                    
steth_20190821_10_29_45.wav                                                                                                                                    
steth_20191017_13_48_43.wav                                                                                                                                    
steth_20190815_14_18_44.wav                                                                                                                                    
steth_20190821_11_03_59.wav                                                                                                                                    
steth_20190809_12_02_54.wav                                                                                                                                    
steth_20190908_10_20_39.wav                                                                                                                                    
steth_20190922_16_56_55.wav                                                                                                                                    
steth_20190815_09_56_36.wav                                                                                                                                    
steth_20190821_10_14_32.wav                                                                                                                                    
steth_20190929_10_02_30.wav

steth_20190821_10_14_10.wav                       
steth_20190809_11_02_39.wav                       
steth_20190809_12_03_44.wav                       
steth_20190810_12_38_05.wav                       
steth_20190815_09_54_14.wav                       
steth_20191001_00_32_24.wav                       
steth_20190809_11_03_38.wav                       
steth_20191027_12_03_38.wav                       
steth_20191025_02_26_57.wav                       
steth_20191001_00_33_09.wav                       
steth_20190815_09_57_45.wav                       
steth_20190902_13_20_20.wav                       
steth_20190821_10_11_22.wav
'''

import sys
file = sys.argv[1]
p = sys.argv[2]

# Set up plot
fig, (ax_raw, ax_spect, ax_base, ax_prop, ax_ground) = plt.subplots(5, 1, sharex=True, figsize=(8, 6))
fig.suptitle(file)

# Remove x axis labels
ax_raw.xaxis.set_visible(False)
ax_spect.xaxis.set_visible(False)
ax_base.xaxis.set_visible(False)
ax_prop.xaxis.set_visible(False)
ax_ground.xaxis.set_visible(False)

# Remove y axis labels
ax_raw.yaxis.set_visible(False)
ax_spect.yaxis.set_visible(False)
ax_base.yaxis.set_visible(False)
ax_prop.yaxis.set_visible(False)
ax_ground.yaxis.set_visible(False)

# Get raw audio
data, sr = librosa.load(folder+file)
t = np.linspace(0, len(data) / sr, num=len(data))
ax_raw.plot(t, data, color="grey", alpha=0.5, zorder=0, label="Audio")

# Get spectrogram
y, spect = load_sample(folder+file)
#spect = librosa.power_to_db(spect)
img = librosa.display.specshow(spect.transpose(),
    x_axis='time',
    y_axis='mel', sr=8000, hop_length=80,
    ax=ax_spect,
    cmap="viridis",
)

def draw(y, ax):
    # Get events
    events = []
    for i in range(4):
        st = -1
        for j in range(len(y)):
            if y[j, i] > 0.4:
                if st == -1:
                    st = j
            else:
                if st != -1:
                    events.append([mapping[i], st, j])
                    st = -1
        if st != -1:
            events.append([mapping[i], st, len(y)])
    print(events)
    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 80 / 8000
        end = (end-1) * 80 / 8000
        ax.axvspan(
            start,
            end,
            vert_size*colors[label][1],
            vert_size*(colors[label][1]+1),
            alpha=0.6,
            color=colors[label][0],
            zorder = 10,
            label=label
        )


# Load proposed model
print("Loading proposed model...", end="")

model = TwinNetwork(64, 4, "cpu", gru_layers=2)

checkpoint = torch.load(f"models/model_proposed_{p}_2.pt")

# Remove all the "module." from the keys because that's there for some reason
checkpoint_clean = {}
for i in checkpoint.keys():
    checkpoint_clean[i.replace("module.", "")] = checkpoint[i]

model.load_state_dict(checkpoint_clean)
print("done")

# run model
spect = spect[None,...]
data = torch.from_numpy(spect).float()
outputs, _ = model(data)
outputs = outputs.detach().numpy()[0]
#outputs = outputs["framewise_output"].detach().numpy()[0]

draw(outputs, ax_prop)

print("Loading baseline model...", end="")

model = TwinNetwork(64, 4, "cpu", gru_layers=2)

checkpoint = torch.load(f"models/model_baseline4_{p}_1.pt")

# Remove all the "module." from the keys because that's there for some reason
checkpoint_clean = {}
for i in checkpoint.keys():
    checkpoint_clean[i.replace("module.", "")] = checkpoint[i]

model.load_state_dict(checkpoint_clean)

print("done")

# run model
data = torch.from_numpy(spect).float()
outputs, _ = model(data)
outputs = outputs.detach().numpy()[0]
#outputs = outputs["framewise_output"].detach().numpy()[0]

draw(outputs, ax_base)

# Draw ground truth

draw(y, ax_ground)

plt.xlim([0, 15])

'''
handles, labels = ax_prop.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_prop.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)

handles, labels = ax_ground.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_ground.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)
'''

print("Saving...")
plt.savefig(f"paper_{p}_"+file[:-4]+".png")
plt.show()