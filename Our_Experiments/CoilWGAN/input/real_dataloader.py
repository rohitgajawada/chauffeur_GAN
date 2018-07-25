import os
import glob
import h5py
import traceback
import sys
import numpy as np
from skimage import io

from torch.utils.data import Dataset
import torch
from logger import coil_logger
from random import shuffle
from torchvision import transforms

class RealDataset():
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, batch_size, transform=None):  # The transformation object.

        self.real_transform = transform
        self.real_image_dir = root_dir
        self.real_image_paths = os.listdir(self.real_image_dir)
        shuffle(self.real_image_paths)
        self.len_real = len(self.real_image_paths)
        self.real_counter = 0
        self.batch_size = batch_size


    def get_imgs(self):

        if self.real_counter >= self.len_real:
            self.real_counter = 0
            shuffle(self.real_image_paths)

        imgs_b = torch.FloatTensor([])
        for i in range(self.batch_size):

            real_img_name = self.real_image_paths[self.real_counter]
            real_img = io.imread(os.path.join(self.real_image_dir, real_img_name))
            real_img = real_img.transpose((2, 0, 1))
            real_img = torch.from_numpy(real_img / 255.0).contiguous()
            real_img = real_img.type(torch.FloatTensor)
            if self.real_transform is not None:
                real_img = self.real_transform(real_img)
            self.real_counter += 1

            image_b = real_img.unsqueeze(0)
            imgs_b = torch.cat((imgs_b, image_b), 0)

        return imgs_b
