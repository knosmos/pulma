import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import h5py

np.random.seed(4)

class HF_Lung_Dataset(Dataset):
    def __init__(self, dataFName, train = True, augment = False):
        self.ds = h5py.File(dataFName, 'r')
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.ds['fname'])

    def __getitem__(self, idx):
        data = self.ds['input'][idx]
        labels = self.ds['gt'][idx]
        if self.augment:
            # time shift augmentation
            #shift = np.random.randint(len(labels))
            shift = np.random.randint(-100, 100)
            data = np.roll(data, shift, axis=0)
            labels = np.roll(labels, shift, axis=0)

            '''
            # random crop
            size = 200
            crop = np.random.randint(0, size)
            data = data[crop:crop + 1500 - size]
            labels = labels[crop:crop + 1500 - size]
            '''
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()