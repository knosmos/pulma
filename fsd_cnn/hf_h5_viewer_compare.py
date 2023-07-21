import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from tqdm import tqdm
import librosa
import preprocess
import os

def load_sample(filename, sr):
    hop = 80

    # load audio
    audio = librosa.load(filename, sr=4000)[0]

    # filter
    audio = preprocess.filter_audio(audio)

    # resample
    audio = librosa.resample(audio, orig_sr=4000, target_sr=sr)

    # extract spec
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, n_fft=256, hop_length=hop)
    spec = librosa.power_to_db(spec)
    spec = np.transpose(spec)

    # load annotations
    #print(f'filename = {filename}')
    with open(filename[:-4] + "_label.txt", "r") as f:
        lines = f.readlines()
        labels = [None] * len(lines)
        for i, line in enumerate(lines):
            label, start, end = line.split()
            labels[i] = [float(start.split(":")[-1]), float(end.split(":")[-1]), label]
    #print(f'labels = {labels}')
    
    # discretize labels and map to spec
    y = np.zeros((spec.shape[0], 6))
    for i in range(spec.shape[0]):
        for label in labels:
            if label[0] <= i * hop / sr <= label[1]:
                if label[2] == "I":
                    y[i, 0] = 1
                elif label[2] == "E":
                    y[i, 1] = 1
                elif label[2] == "D":
                    y[i, 2] = 1
                elif label[2] == "Rhonchi":
                    y[i, 3] = 1
                elif label[2] == "Wheeze":
                    y[i, 4] = 1
                elif label[2] == "Stridor":
                    y[i, 5] = 1

    #print(f'labelsum = {sum(y)}')
    return y, spec

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

def display4k(idx, spect, labels):
    # Set up figure
    fig, (ax_spect, ax_ground) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Spectrogram
    img = librosa.display.specshow(spect.transpose(),
        x_axis='time',
        y_axis='mel',
        sr=4000,
        ax=ax_spect,
        cmap="viridis",
        hop_length = 80,
        n_fft = 256
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

    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 80 / 4000
        end = (end-1) * 80 / 4000
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

    handles, labels = ax_ground.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_ground.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)
    plt.savefig(f"h5viz_comp/{idx}_4k.png")
    plt.close()

def display8k(idx, spect, labels):
    # Set up figure
    fig, (ax_spect, ax_ground) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Spectrogram
    img = librosa.display.specshow(spect.transpose(),
        x_axis='time',
        y_axis='mel',
        sr=8000,
        ax=ax_spect,
        cmap="viridis",
        hop_length = 80,
        n_fft = 256
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

    # draw events
    vert_size = 1/len(colors)
    for line in events:
        label, start, end = line
        start = start * 80 / 8000
        end = (end-1) * 80 / 8000
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

    handles, labels = ax_ground.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_ground.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)
    plt.savefig(f"h5viz_comp/{idx}_8k.png")
    plt.close()

for file in tqdm(os.listdir("../data/train/")):
    if not file.endswith(".wav"):
        continue
    y, spec = load_sample("../data/train/" + file, 4000)
    display4k(file, spec, y)
    y, spec = load_sample("../data/train/" + file, 8000)
    display8k(file, spec, y)