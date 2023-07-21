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

labels = ["I", "E", "D", "Rhonchi", "Wheeze", "Stridor"]
sounds_curr = {label: np.array([]) for label in labels}

k = 0

for file in os.listdir("data/train"):
    if not file.endswith(".wav"):
        continue

    spf = wave.open("data/train/"+file, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    fs = spf.getframerate()

    # Get annotation
    with open("data/train/"+file[:-4]+"_label.txt", "r") as f:
        lines = f.readlines()
        vert_size = 1/len(colors)
        for line in lines:
            label, start, end = line.split()
            start = float(start.split(":")[-1]) * fs
            end = float(end.split(":")[-1]) * fs
            sounds_curr[label] = np.append(sounds_curr[label], signal[int(start):int(end)])

    k += 1
    print(file, k)
    if k > 100:
        break

for label in labels:
    wave_out = wave.open("sound_samples/"+label+".wav", "w")
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(fs)
    wave_out.writeframes(sounds_curr[label].astype(np.int16).tobytes())
    wave_out.close()