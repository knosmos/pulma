import torch
from torch.utils.data import Dataset
from dataset_gen import load_sample
import os
import glob
import numpy as np
from tqdm import tqdm
import h5py

class HF_Lung_Dataset(Dataset):
    def __init__(self, dataFName, train = True):
        self.ds = h5py.File(dataFName, 'r') 

    def __len__(self):
        return len(self.ds['input'])

    def __getitem__(self, idx):
        data = self.ds['input'][idx]
        labels = self.ds['gt'][idx]
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()
