import matplotlib.pyplot as plt
import h5py
import librosa.display
from tqdm import tqdm

ds = h5py.File("train_fsd.h5", 'r')

plt.style.use("ggplot")

def display(idx, spect):
    # Set up figure
    fig, (ax_spect) = plt.subplots(1, 1, sharex=True, figsize=(5, 3))

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

    plt.xlim([0, 1])
    plt.savefig(f"fsdviz/{idx}.png")
    plt.close()

for idx in tqdm(range(100)):
    display(idx, ds["input"][idx])