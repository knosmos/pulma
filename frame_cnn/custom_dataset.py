import torch
from torch.utils.data import Dataset
from dataset_gen import load_sample
import os
import numpy as np
from tqdm import tqdm
import glob

class HF_Lung_Dataset(Dataset):
    def __init__(self, train = True):
        self.data = None
        self.labels = None

        #self.cap = 5000
        #self.cap = 2500
        self.cap = 1000

        if train:
            path = "../data/train/"
        else:
            path = "../data/test/"

        print("Loading data from: " + path)
        #files = glob.glob(os.path.join(path, '*L1*.wav'))
        files = glob.glob(os.path.join(path, '*.wav'))
        #for filename in tqdm(os.listdir(path)[:self.cap]):
        for filename in tqdm(files[:self.cap]):
            if filename.endswith(".wav"):
                #labels, data = load_sample(path + filename)
                labels, data = load_sample(filename)
                #print(f'{filename}: datasize = {data.shape}, labelsize={labels.shape}')
                if self.data is None:
                    self.data = data
                    self.labels = labels
                else:
                    self.data = np.concatenate((self.data, data))
                    self.labels = np.concatenate((self.labels, labels))
                #print(f'accumuated datasize = {self.data.shape}, labelsize={self.labels.shape}')
        print(f"Loaded {len(self.data)} samples")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.from_numpy(self.labels[idx]).float()
