import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import numpy as np
import os, glob

colors = {
    "I": ["crimson", 0], # Inhalation
    "E": ["cornflowerblue", 1], # Exhalation
    "Crackle": ["mediumseagreen", 2], # Discontinuous adventitious sound?
    "Rhonchi": ["slategrey",3],
    "Wheeze": ["indigo",4],
    "Stridor": ["orange",5]
}
plt.style.use("ggplot")

k = 0
ctr = 0

print(len(glob.glob(os.path.join("data/train", "steth_*.wav"))))

for file in sorted(glob.glob(os.path.join("data/train", "steth_*.wav"))):
    if not file.endswith(".wav"):
        continue

    if ctr != 2518:
        ctr += 1
        continue

    # Set up plot
    fig, (ax_raw, ax_spect, ax_annot) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    fig.suptitle(file)

    # Remove x axis labels
    ax_raw.xaxis.set_visible(False)
    ax_spect.xaxis.set_visible(False)
    ax_annot.xaxis.set_visible(False)

    # Remove y axis labels
    ax_raw.yaxis.set_visible(False)
    ax_spect.yaxis.set_visible(False)
    ax_annot.yaxis.set_visible(False)

    # Get raw audio
    data, sr = librosa.load(file)
    t = np.linspace(0, len(data) / sr, num=len(data))
    ax_raw.plot(t, data, color="grey", alpha=0.5, zorder=0, label="Audio")

    # Get spectrogram
    spect = librosa.feature.melspectrogram(y=data, sr=sr)

    spect_dB = librosa.power_to_db(spect, ref=np.max)

    img = librosa.display.specshow(spect_dB,
        x_axis='time',
        y_axis='mel', sr=sr,
        ax=ax_spect,
        cmap="viridis",
    )

    # Get annotation
    with open(file[:-4]+"_label.txt", "r") as f:
        lines = f.readlines()
        print(lines)
        vert_size = 1/len(colors)
        for line in lines:
            label, start, end = line.split()
            start = float(start.split(":")[-1])
            end = float(end.split(":")[-1])
            #if label in "IE":
            #    continue
            if label == "D":
                label = "Crackle"
            plt.axvspan(
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

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 12}).set_zorder(1000)
    plt.savefig("inhale.png")
    #plt.show()
    plt.clf()

    k += 1
    print(file, k)

    break