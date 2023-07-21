import librosa
import os
import soundfile as sf

for i, filename in enumerate(os.listdir("../FSD50k/FSD50K.dev_audio/")):
    # load audio
    audio, sr = librosa.load("../FSD50k/FSD50K.dev_audio/" + filename)

    # resample
    audio = librosa.resample(audio, orig_sr=sr, target_sr=4000)

    # save
    sf.write("fsdviz/" + filename, audio, 4000)

    if i > 10:
        break