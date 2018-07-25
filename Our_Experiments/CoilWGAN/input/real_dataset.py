import os
import glob
import h5py
import traceback
import sys
import numpy as np

from torch.utils.data import Dataset
import torch

from configs import g_conf

class SynthDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.img_size = [3, 88, 200]
        self.image_paths = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_name = self.image_paths[idx]
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img / 1.0).contiguous()
        img = self.transform(img)
        return img
