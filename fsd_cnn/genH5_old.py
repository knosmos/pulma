import torch
from torch.utils.data import Dataset
from dataset_gen import load_sample
import os
import glob
import numpy as np
from tqdm import tqdm
import h5py

def save_h5(h5fName, data, labels, filenames):
    inputs = np.array(data, dtype=np.float32)   
    gts = np.array(labels, dtype=np.float32)
    fnames = np.array(filenames, dtype=np.string_)
    with h5py.File(h5fName, 'w') as ds:
        ds.create_dataset("input", data=inputs)
        ds.create_dataset("gt", data=gts)
        ds.create_dataset("fname", data=fnames)

def genH5():
        path = "../data/train/"
        print("Loading data from: " + path)
        #files = glob.glob(os.path.join(path, '*L1*.wav'))
        files = glob.glob(os.path.join(path, "steth_*.wav"))
        numFiles = len(files)
        accdata = []
        acclabels = []
        accfilenames = []
        for filename in tqdm(sorted(files)):
            #print(filename.split("/")[-1])
            accfilenames.append(filename.split("/")[-1])
            labels, data = load_sample(filename)
            accdata.append(data)
            acclabels.append(labels)
            print('.', end='', flush=True)
        save_h5('train_steth.h5', accdata, acclabels, accfilenames)

        path = "../data/test/"
        print("Loading data from: " + path)
        #files = glob.glob(os.path.join(path, '*L1*.wav'))
        files = glob.glob(os.path.join(path, "steth_*.wav"))
        numFiles = len(files)
        accdata = []
        acclabels = []
        accfilenames = []
        for filename in tqdm(sorted(files)):
            #print(filename.split("/")[-1])
            accfilenames.append(filename.split("/")[-1])
            labels, data = load_sample(filename)
            accdata.append(data)
            acclabels.append(labels)
            print('.', end='', flush=True)
        save_h5('test_steth.h5', accdata, acclabels, accfilenames)


genH5()
