import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from tqdm import tqdm
import h5py
import librosa
import numpy as np
import csv
from multiprocessing import Pool
import preprocess

# Generate dictionary of labels --> indices
vocab = {}
def build_vocabulary():
    with open("../FSD50k/FSD50K.ground_truth/vocabulary.csv") as f:
        for line in f.read().split("\n"):
            if len(line):
                idx, label, id = line.split(",")
                vocab[id] = int(idx)

build_vocabulary()
vocab_size = len(vocab)
print("Loaded vocabulary:", vocab_size, "classes")

size = 4000 # size of each slice (1 second)

# Get audio spectrogram
def load_sample(filename):
    # load audio
    audio, sr = librosa.load(filename)

    # resample
    audio = librosa.resample(audio, orig_sr=sr, target_sr=4000)
    audio = preprocess.filter_audio(audio)

    # slice or repeat as necessary
    if len(audio) < size:
        repeats = size // len(audio) + 1
        audios = [np.tile(audio, repeats)[:size]]
    else:
        audios = []
        hopsize = int(0.5 * 4000)
        for i in range(0, len(audio), hopsize):
            piece = audio[i:i + size]
            repeats = size // len(piece) + 1
            audios.append(np.tile(piece, repeats)[:size])

    # extract spec
    specs = []
    for y in audios:
        spec = librosa.feature.melspectrogram(y=y, sr=4000, n_mels=64, n_fft=256, hop_length=80)
        spec = librosa.power_to_db(spec)
        spec = np.transpose(spec)
        #print(spec.shape)
        specs.append(spec)
    
    return specs

def save_h5(h5fName, data, labels, filenames=None):
    inputs = np.array(data, dtype=np.float32)
    gts = np.array(labels, dtype=np.float32)
    #fnames = np.array(filenames, dtype=np.string_)
    with h5py.File(h5fName, 'w') as ds:
        ds.create_dataset("input", data=inputs)
        ds.create_dataset("gt", data=gts)
        #ds.create_dataset("fname", data=fnames)

def process(idx):
    filename = filenames[idx]
    mids = ids[idx]
    cat = split[idx]
    data = load_sample(path + filename + ".wav")
    labels = np.zeros(vocab_size)
    for mid in mids.split(","):
        labels[vocab[mid]] = 1
    if idx % 500 == 0:
        print(f"completed idx {idx}")
    return [data, labels, cat]

def genH5():
    global filenames, ids, path, split

    path = "../FSD50k/FSD50K.dev_audio/"
    print("Loading data from: " + path)
    split = []

    with open("../FSD50k/FSD50K.ground_truth/dev.csv") as f:
        filenames = []
        ids = []
        # read tasks
        for line in csv.reader(f):
            filename, _, mids, cat = line
            filenames.append(filename)
            ids.append(mids)
            split.append(cat) # train/val
        filenames.pop(0)
        ids.pop(0)
    # process
    with Pool(32) as p:
        r = p.map(process, range(len(filenames)))
    # squash
    tdata = []
    tlabels = []
    vdata = []
    vlabels = []
    for data, labels, cat in r:
        if cat == "train":
            tdata += data
            for i in range(len(data)):
                tlabels.append(labels)
        if cat == "val":
            vdata += data
            for i in range(len(data)):
                vlabels.append(labels)

    # save
    save_h5('train_fsd.h5', tdata, tlabels)
    save_h5('val_fsd.h5', vdata, vlabels)

    # reset
    path = "../FSD50k/FSD50K.eval_audio/"
    print("Loading data from: " + path)
    accdata = []
    acclabels = []
    accfilenames = []
    with open("../FSD50k/FSD50K.ground_truth/eval.csv") as f:
        filenames = []
        ids = []
        # read tasks
        for line in csv.reader(f):
            filename, _, mids = line
            filenames.append(filename)
            ids.append(mids)
        filenames.pop(0)
        ids.pop(0)
    # process
    with Pool(32) as p:
        r = p.map(process, range(len(filenames)))
    # squash
    for data, labels, _ in r:
        accdata += data
        for i in range(len(data)):
            acclabels.append(labels)
    # save
    save_h5('test_fsd.h5', accdata, acclabels)


    '''
    # Without multiprocessing

    with open("../FSD50k/FSD50K.ground_truth/dev.csv") as f:
        for line in tqdm(csv.reader(f), total=40967):
            if not len(line) or line[0] == "fname": continue
            filename, _, ids, _ = line
            accfilenames.append(filename)
            data = load_sample(path + filename + ".wav")
            labels = np.zeros(vocab_size)
            for id in ids.split(","):
                labels[vocab[id]] = 1
            accdata += data
            for i in range(len(data)):
                acclabels.append(labels)
            if (len(accdata)) > 100:
                break
    save_h5('train_fsd.h5', accdata, acclabels, accfilenames)

    path = "../FSD50k/FSD50K.eval_audio/"
    print("Loading data from: " + path)
    accdata = []
    acclabels = []
    accfilenames = []
    with open("../FSD50k/FSD50K.ground_truth/eval.csv") as f:
        for line in tqdm(csv.reader(f), total=10232):
            if not len(line) or line[0] == "fname": continue
            filename, _, ids = line
            accfilenames.append(filename)
            data = load_sample(path + filename + ".wav")
            labels = np.zeros(vocab_size)
            for id in ids:
                labels[vocab[id]] = 1
            accdata += data
            for i in range(len(data)):
                acclabels.append(labels)
    save_h5('test_fsd.h5', accdata, acclabels, accfilenames)
    '''



genH5()