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
from dataset_gen import load_sample

def save_h5(h5fName, data, labels, filenames=None):
    inputs = np.array(data, dtype=np.float32)
    gts = np.array(labels, dtype=np.float32)
    #fnames = np.array(filenames, dtype=np.string_)
    with h5py.File(h5fName, 'w') as ds:
        ds.create_dataset("input", data=inputs)
        ds.create_dataset("gt", data=gts)
        #ds.create_dataset("fname", data=fnames)

def genH5():
    global filenames, ids, path, split

    path = "../data/train/"
    print("Loading data from: " + path)

    filenames = glob.glob(os.path.join(path, "steth_*.wav"))

    # process
    with Pool(32) as p:
        r = p.map(load_sample, filenames)

    # zip
    accdata = []
    acclabels = []
    for labels, data in r:
        accdata.append(data)
        acclabels.append(labels)

    # save
    save_h5('train_hf.h5', accdata, acclabels)

    # reset
    path = "../data/test/"
    print("Loading data from: " + path)
    filenames = glob.glob(os.path.join(path, "steth_*.wav"))

    # process
    with Pool(32) as p:
        r = p.map(load_sample, filenames)

    # zip
    accdata = []
    acclabels = []
    for labels, data in r:
        accdata.append(data)
        acclabels.append(labels)

    # save
    save_h5('test_hf.h5', accdata, acclabels)

genH5()