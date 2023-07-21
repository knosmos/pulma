'''
Preprocesssing steps:
- High pass filter (100hz)
- Low pass filter (1000hz)
- Noise reduction (time mask smoothing 64ms)
- Denoising (wavelet)
'''

import librosa
import librosa.display
import numpy as np
import os
from tqdm import tqdm
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr
from skimage.restoration import denoise_wavelet

colors = {
    "I": ["crimson", 0], # Inhalation
    "E": ["cornflowerblue", 1], # Exhalation
    "D": ["mediumseagreen", 2], # Discontinuous adventitious sound?
    "Rhonchi": ["slategrey",3],
    "Wheeze": ["indigo",4],
    "Stridor": ["orange",5]
}
mapping = ["I", "E", "D", "Rhonchi", "Wheeze", "Stridor"]
#plt.style.use("ggplot")
plt.rcParams.update({'font.size': 6})

def load_audio(filename):
    # load audio
    audio = librosa.load(filename, sr=4000)[0]
    return audio

def butter_highpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=10):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def filter_audio(audio):
    #print(filename, audio.shape)
    sr = 4000
    filtered = butter_highpass_filter(audio, 100, sr)
    #filtered = butter_lowpass_filter(filtered, 1000, sr)

    orig_shape = filtered.shape
    filtered = np.reshape(filtered, (2, -1))

    filtered = nr.reduce_noise(y=filtered, sr=sr,  time_mask_smooth_ms=64, prop_decrease=0.90)
    filtered = filtered.reshape(orig_shape)

    #filtered = denoise_wavelet(filtered, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')

    # normalize
    filtered = filtered / np.max(np.abs(filtered))
    
    # resample
    filtered = librosa.resample(filtered, orig_sr=sr, target_sr=8000)

    return filtered


if __name__ == "__main__":
    # load
    folder = "../data/test/"
    for file in tqdm(os.listdir(folder)):
        if not file.endswith(".wav"):
            continue

        audio = load_audio(folder + file)

        # filter
        filtered = filter_audio(audio)
        
        # save
        sf.write('filtered.wav', filtered, 32000)

        # plot
        plt.figure(figsize=(8,6))
        plt.suptitle(file)

        ax1 = plt.subplot(511)
        t = np.linspace(0, len(audio) / 4000, num=len(audio))
        plt.plot(t, audio)

        t2 = np.linspace(0, len(filtered) / 8000, num=len(filtered))
        ax2 = plt.subplot(512, sharex=ax1)
        plt.plot(t2, filtered)

        ax3 = plt.subplot(513, sharex=ax2)
        librosa.display.specshow(
            librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=4000, n_mels=64, n_fft=1024, hop_length=320),
                ref=np.max,
            ),
            x_axis='time',
            y_axis='mel',
            ax=plt.subplot(513),
            cmap="viridis",
            sr=4000,
            hop_length=320
        )

        ax4 = plt.subplot(514, sharex=ax3)
        librosa.display.specshow(
            librosa.power_to_db(
                librosa.feature.melspectrogram(y=filtered, sr=8000, n_mels=64, n_fft=1024, hop_length=320),
                ref=np.max,
            ),
            x_axis='time',
            y_axis='mel',
            ax=plt.subplot(514),
            cmap="viridis",
            sr=32000,
            hop_length=320
        )

        ax5 = plt.subplot(515, sharex=ax4)
        with open(folder+file[:-4]+"_label.txt", "r") as f:
            lines = f.readlines()

        vert_size = 1/len(colors)
        for line in lines:
            label, start, end = line.split()
            start = float(start.split(":")[-1])
            end = float(end.split(":")[-1])
            ax5.axvspan(
                start,
                end,
                vert_size*colors[label][1],
                vert_size*(colors[label][1]+1),
                alpha=0.6,
                color=colors[label][0],
                zorder = 10,
                label=label
            )

        ax1.xaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        ax3.xaxis.set_visible(False)
        ax4.xaxis.set_visible(False)

        plt.show()
        #plt.savefig("data_vis/"+file[:-4]+".png")

        break
