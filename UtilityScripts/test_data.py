#############################################
# Script for testing the recorded .h5 files #
# For debugging purposes only.              #
#############################################

import h5py
from PIL import Image
import numpy as np
import glob
import time
import os
import time
import matplotlib.pyplot as plt
import random



low_files = glob.glob('/home/adas/Documents/CARLA_0.8.2_SantQuirze/PythonClient/_out16_Carla__exp1/*.h5')
# low_files = glob.glob('/home/adas/Documents/VirtualElektraData2_AllWeather/SeqTrain/*.h5')
# for efile, lfile in zip(epic_files, low_files):
# random.shuffle(low_files)

print (low_files)
for lfile in low_files:
    print("Opening::", lfile)
    # edata = h5py.File(efile, 'r+')
    ldata = h5py.File(lfile, 'r+')

    for i in xrange(0, len(ldata['rgb']), 5):

        #x = random.randint(1, 100)
        #if x < 5:
        #    continue

        low_imgframe = ldata['rgb'][i]
        low_imgframe = np.transpose(low_imgframe,(1,2,0))
        float_data = ldata['targets']
	print(float_data[i][0])
        print ("Steering", float_data[i + 4][0], float_data[i + 3][0], float_data[i][0], float_data[i + 1][0], float_data[i + 2][0]), "Camera Angle", float_data[i][26])
        
        # For checking images at turns 
        # if(float_data[i][0] < 0.7 and float_data[i][0] > -0.7):
        #     continue

        plt.figure(1)
        plt.subplot(511)
        plt.imshow(Image.fromarray((np.uint8(ldata['rgb'][i+4]*255))))
        plt.subplot(512)
        plt.imshow(Image.fromarray((np.uint8(ldata['rgb'][i+3]*255))))
        # plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+3]*255), (1,2,0)))
        plt.subplot(513)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i]*255)))
        plt.subplot(514)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+1]*255)))
        plt.subplot(515)
        plt.imshow(Image.fromarray(np.uint8(ldata['rgb'][i+2]*255)))
        plt.show()
        
