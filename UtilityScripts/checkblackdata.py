import h5py
from PIL import Image
import numpy as np
import glob
import time
import os
import time
import matplotlib.pyplot as plt
import random


low_files = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/July10_UNIT_W12/*.h5')
low_files.sort(reverse=True)

print (low_files)
for lfile in low_files:
    print("Opening::", lfile)
    ldata = h5py.File(lfile, 'r+')
    i = 0

    low_imgframe = ldata['rgb'][i]
    low_imgframe = np.transpose(low_imgframe,(1,2,0))
    float_data = ldata['targets']

    if ldata['rgb'][i+1][0][0][0] == 0 and  ldata['rgb'][i+1][0][0][1] == 0 and ldata['rgb'][i+1][0][0][2] == 0:
        plt.figure(1)
        plt.subplot(511)
        plt.imshow(Image.fromarray((np.uint8(ldata['rgb'][i]*255))))
        plt.subplot(512)
        plt.imshow(Image.fromarray((np.uint8(ldata['rgb'][i+1]*255))))

        plt.subplot(513)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+2]*255)))
        plt.subplot(514)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+3]*255)))
        plt.subplot(515)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+4]*255)))
        plt.show()
