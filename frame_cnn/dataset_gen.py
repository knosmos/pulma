import librosa
import numpy as np

def load_sample(filename):
    # load audio
    audio = librosa.load(filename, sr=4000)[0]

    # extract spec
    spec = librosa.feature.melspectrogram(y=audio, sr=4000, n_mels=64, n_fft=2048, hop_length=256)
    spec = np.transpose(spec)

    # load annotations
    with open(filename[:-4] + "_label.txt", "r") as f:
        lines = f.readlines()
        labels = [None] * len(lines)
        for i, line in enumerate(lines):
            label, start, end = line.split()
            labels[i] = [float(start.split(":")[-1]), float(end.split(":")[-1]), label]

    # discretize labels and map to spec
    y = np.zeros((spec.shape[0], 6))
    for i in range(spec.shape[0]):
        for label in labels:
            if label[0] <= i * 256 / 4000 <= label[1]:
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
    return y, spec