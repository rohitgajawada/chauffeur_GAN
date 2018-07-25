

###############################################################
# Script for sampling images for UNIT/CycleGAN from .h5 files #
###############################################################


import h5py
from PIL import Image
import numpy as np
import glob
import time
import os
import matplotlib.pyplot as plt
from random import shuffle

epic_files= glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/EpicImageDataset/*.png')
epicimg_path = '/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/EpicSampled/'

# lowimgtest = '/home/adas/Documents/pytorch-CycleGAN-and-pix2pix/datasets/low2epic/testA/'
# epicimg_path = '/home/adas/Documents/pytorch-CycleGAN-and-pix2pix/datasets/low2epic/trainB/'
# epicimgtest = '/home/adas/Documents/pytorch-CycleGAN-and-pix2pix/datasets/low2epic/testB/'
# epic_files = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/EpicImageDataset/*.png')

shuffle(epic_files)
# shuffle(low_files)

train_samples = 41000
# test_samples = 300

epic_train = epic_files[:train_samples]
# low_train = low_files[:train_samples]
# epic_test = epic_files[train_samples: train_samples + test_samples]
# low_test = low_files[train_samples: train_samples + test_samples]

print("train")
for i in range(train_samples):
    # a_file = low_train[i]
    b_file = epic_train[i]
    # os.rename(a_file, lowimg_path + str(i) + '.png')
    os.rename(b_file, epicimg_path + str(i) + '.png')

# print("test")
# for i in range(test_samples):
#     a_file = low_test[i]
#     b_file = epic_test[i]
#     os.rename(a_file, lowimgtest + str(i) + '.png')
#     os.rename(b_file, epicimgtest + str(i) + '.png')
