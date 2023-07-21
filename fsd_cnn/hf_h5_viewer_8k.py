import matplotlib.pyplot as plt
import h5py
import librosa.display
from tqdm import tqdm

ds = h5py.File("train_steth.h5", 'r')

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

def display(idx, spect, labels):
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
    plt.savefig(f"h5viz8k/{idx}.png")
    plt.close()

for idx in tqdm(range(len(ds["input"]))):
    display(idx, ds["input"][idx], ds["gt"][idx])