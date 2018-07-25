###############################################################
# Script for creating images for UNIT/CycleGAN from .h5 files #
###############################################################

import h5py
from PIL import Image
import numpy as np
import glob
import time
import os
import matplotlib.pyplot as plt
from random import shuffle
import random

epicimg_path = './lala/'

epic_files = glob.glob('./data_00000.h5')
epic_files.sort()

ct = -1
val = 0
for efile in epic_files:
    edata = h5py.File(efile, 'r+')
    ct += 1
    print(ct)
    length = len(edata['rgb'])
    for i in range(length):

        epic_imgframe = edata['rgb'][i]
        epic_img = Image.fromarray(np.uint8(epic_imgframe*255))

        epic_img.save(epicimg_path + str(val) + '.png')

        val += 1

