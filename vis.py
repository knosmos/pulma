import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

colors = {
    "I": ["red", 0], # Inhalation
    "E": ["blue", 1], # Exhalation
    "D": ["green", 2], # Discontinuous adventitious sound?
    "Rhonchi": ["black",3],
    "Wheeze": ["purple",4],
    "Stridor": ["orange",5]
}

plt.style.use("ggplot")

spf = wave.open("data/"+sys.argv[1]+".wav", "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.frombuffer(signal, np.int16)
fs = spf.getframerate()

time = np.linspace(0, len(signal) / fs, num=len(signal))

plt.figure(1)
plt.title(sys.argv[1])
plt.plot(time, signal, color="grey", alpha=0.5, zorder=0, label="Audio")

# Get annotation
with open("data/"+sys.argv[1]+"_label.txt", "r") as f:
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
            alpha=0.5,
            color=colors[label][0],
            zorder = 10,
            label=label
        )

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()