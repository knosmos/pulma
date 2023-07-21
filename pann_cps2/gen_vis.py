import matplotlib.pyplot as plt
import numpy as np
import wave
import os

colors = {
    "I": ["crimson", 0], # Inhalation
    "E": ["cornflowerblue", 1], # Exhalation
    "D": ["mediumseagreen", 2], # Discontinuous adventitious sound?
    "Rhonchi": ["slategrey",3],
    "Wheeze": ["indigo",4],
    "Stridor": ["orange",5]
}
plt.style.use("ggplot")

k = 0

for file in os.listdir("data/train"):
    if not file.endswith(".wav"):
        continue

    spf = wave.open("data/train/"+file, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    fs = spf.getframerate()

    time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title(file)
    plt.plot(time, signal, color="grey", alpha=0.5, zorder=0, label="Audio")

    # Get annotation
    with open("data/train/"+file[:-4]+"_label.txt", "r") as f:
        lines = f.readlines()
        vert_size = 1/len(colors)
        for line in lines:
            label, start, end = line.split()
            start = float(start.split(":")[-1])
            end = float(end.split(":")[-1])
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

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", prop={'size': 6}).set_zorder(1000)
    plt.savefig("vis/"+file[:-4]+".png")
    plt.clf()

    k += 1
    print(file, k)
    #if k > 100:
    #    break