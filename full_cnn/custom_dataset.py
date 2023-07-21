import torch
from torch.utils.data import Dataset
from dataset_gen import load_sample
import os
import glob
import numpy as np
from tqdm import tqdm

class HF_Lung_Dataset(Dataset):
    def __init__(self, train = True):
        self.cap = 1000
        
        #self.data = np.zeros(shape = (self.cap, 938, 128))
        #self.labels = np.zeros(shape = (self.cap, 938, 6))

        if train:
            self.path = "../data/train/"
        else:
            self.path = "../data/test/"

        print("Loading data from: " + self.path)
        i = 0

        self.filenames = []
        files = glob.glob(os.path.join(self.path, '*L1*.wav'))
        for filename in tqdm(files):
            if filename.endswith(".wav"):
                if i >= self.cap:
                    break
                #print(filename.split("/")[-1])
                self.filenames.append(filename.split("/")[-1])
                #self.data[i] = data
                #self.labels[i] = labels
                i += 1
        print(f"Loaded {len(self.filenames)} samples")
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if os.path.exists("cache/" + filename[:-4] + "_d.npy"):
            data = np.load("cache/" + filename[:-4] + "_d.npy", allow_pickle=True)
            labels = np.load("cache/" + filename[:-4] + "_l.npy", allow_pickle=True)
        else:
            labels, data = load_sample(self.path + filename)
            np.save("cache/" + filename[:-4] + "_d.npy", data)
            np.save("cache/" + filename[:-4] + "_l.npy", labels)
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()